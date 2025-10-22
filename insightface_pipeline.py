#!/usr/bin/env python3
"""
Cluster faces in an mp4 using InsightFace (RetinaFace detector + ArcFace embeddings),
HDBSCAN clustering, temporal smoothing, and multi-factor quality scoring.

Usage:
    python cluster_video_insightface.py input_video.mp4 output_folder [--gpu]

Requirements (install via pip):
    pip install opencv-python-headless insightface numpy tqdm hdbscan scikit-learn imutils

Notes:
 - GPU: if you have MXNet/onnxruntime/cuda setup for insightface it will use ctx_id=0. The script
   auto-tries CPU if GPU not available.
 - This is streaming-friendly: faces/embeddings are saved to disk during extraction so RAM stays low.
"""
import os
import json
import shutil
import argparse
import tempfile
from tqdm import tqdm
from collections import defaultdict

import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# InsightFace
from insightface.app import FaceAnalysis
# Clustering
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

import crop

print("starting")
# ------------------------
# CONFIGURATION (tweak)
# ------------------------
FRAME_INTERVAL = 20           # process every Nth frame (increase for speed)
MIN_FACE_SIZE = 30           # min face bbox side (px) to keep
RESIZE_WIDTH = 224          # if set, downscale frames to this width for speed (maintain aspect)
MAX_FACES = None             # optional cap for total faces processed (for debug)
BLUR_MIN = 40.0              # min acceptable variance of Laplacian (lower = blurrier)
BRIGHTNESS_MIN = 40.0        # minimum mean brightness acceptable
FACE_SIZE_WEIGHT = 0.25      # weights for quality score components
BLUR_WEIGHT = 0.35
BRIGHTNESS_WEIGHT = 0.15
DETECTION_THRESHOLD = 0.6
QUALITY_THRESHOLD = 1.1

HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 1

# ------------------------
# Helper functions
# ------------------------
def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

model_path = "/home/colet/programming/projects/cast-image-quality/face_quality_resnet50.pth"

# 1. Rebuild the same model architecture
base = models.resnet50(pretrained=False)
base.fc = nn.Linear(base.fc.in_features, 1)
model = base
model.load_state_dict(torch.load(model_path, map_location="cuda"))
model.eval().cuda()

# 2. Define the same transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def crop_face_square(frame, f):
    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, f.bbox.round())
    w = x2 - x1
    h = y2 - y1

    # Skip too-small faces
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None

    # --- Step 1: Make it square ---
    # Find center and max side length
    cx, cy = x1 + w / 2, y1 + h / 2
    side = max(w, h)

    # --- Step 2: Expand square by 30% (or your chosen ratio) ---
    expand_ratio = 0.4
    side = side * (1 + expand_ratio)

    # --- Step 3: Compute new coordinates ---
    x1_new = int(cx - side / 2)
    y1_new = int(cy - side / 2)
    x2_new = int(cx + side / 2)
    y2_new = int(cy + side / 2)

    # --- Step 4: Clip to image boundaries ---
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(frame.shape[1], x2_new)
    y2_new = min(frame.shape[0], y2_new)

    # --- Step 5: Crop and resize ---
    face_crop = frame[y1_new:y2_new, x1_new:x2_new].copy()
    if face_crop.size == 0:
        return None

    face_resized = cv2.resize(face_crop, (RESIZE_WIDTH, RESIZE_WIDTH))
    return face_resized

@torch.no_grad()
def predict_quality(face_resized):
    # Convert NumPy array (BGR from cv2) to PIL Image (RGB)
    img = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).cuda()  # shape: (1, 3, 224, 224)
    score = model(img)
    return score.item()

def select_temporal_candidates(clusters: defaultdict[int, dict], video_duration_sec, num_candidates=3) -> defaultdict[int, list]:
    """
    Select candidate images spanning equal parts of the video.
    
    Args:
        items: dict of label:str -> list[metadata]
        video_duration_sec: total video duration in seconds
        num_candidates: number of candidates to select (default 3)
    
    Returns:
        dict of selected metadata dicts for each cluster, sorted by time
    """

    cluster_candidates = defaultdict(list)
    for label, metadata in clusters.items():
        # skip noise
        if label == -1:
            continue

        if len(metadata) <= num_candidates:
            cluster_candidates[label] = sorted(metadata, key=lambda x: x['time_sec'])
            continue
    
        # Divide video into equal segments (you could implement this by getting the span
        # a person(cluster) appears in a video and find candidates in this span instead)
        segment_duration = video_duration_sec / num_candidates
        
        for segment_idx in range(num_candidates):
            segment_start = segment_idx * segment_duration
            segment_end = (segment_idx + 1) * segment_duration
            
            # Find meta_list in this segment
            segment_meta_list = [
                item for item in metadata
                if segment_start <= item['time_sec'] < segment_end
            ]
            
            if segment_meta_list:
                # Pick the highest quality item from this segment
                best = max(segment_meta_list, key=lambda x: x['quality'])
                cluster_candidates[label].append(best)
        
    return cluster_candidates

# ------------------------
# Main pipeline
# ------------------------
def pipeline(video_path, out_folder, use_gpu=False, verbose=True):
    os.makedirs(out_folder, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="face_cluster_")
    crops_dir = os.path.join(tmp_dir, "crops")
    enc_dir = os.path.join(tmp_dir, "encs")
    frames_dir = os.path.join(tmp_dir, "frames")

    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize face analysis (InsightFace). name "buffalo_l" is robust model (includes ArcFace)
    ctx_id = 0 if use_gpu else -1
    if verbose:
        print("Initializing InsightFace (this will download models on first run)... ctx_id=", ctx_id)
    app = FaceAnalysis(
        name="buffalo_l", 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id < 0 else None)
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video: " + video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if verbose:
        print(f"Video opened: {video_path}, frames={total_frames}, fps={fps}")

    frame_idx = 0
    face_idx = 0
    meta_list = []
    embeddings = []

    pbar = tqdm(total=total_frames // FRAME_INTERVAL + 1, desc="Frames") if verbose else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop.resize_for_detection(frame)
            frame = crop.remove_borders(frame)

            if frame_idx % FRAME_INTERVAL != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_name = f"frame_{frame_idx:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)
            cv2.imwrite(frame_path, frame)


            faces = app.get(rgb)  # returns list of Face objects with bbox, kps, embedding, det_score, pose, etc.

            for f in faces:
                det_score = float(f.det_score) if hasattr(f, "det_score") else 1.0
                if det_score < DETECTION_THRESHOLD:
                    continue

                # f.bbox: [x1,y1,x2,y2], f.kps: landmarks, f.det_score, f.embedding (ArcFace)
                x1, y1, x2, y2 = map(int, f.bbox.round().astype(int))
                # ensure box inside frame
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
                w = x2 - x1; h = y2 - y1
                if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                    continue

                # crop BGR face (for quality scoring & saving)
                face_crop_bgr = crop_face_square(frame, f)
                if face_crop_bgr is None or face_crop_bgr.size == 0:
                    continue

                quality = predict_quality(face_crop_bgr)
                if quality < QUALITY_THRESHOLD:
                    continue

                # Save crop and embedding
                crop_path = os.path.join(crops_dir, f"face_{face_idx}.jpg")
                enc_path = os.path.join(enc_dir, f"enc_{face_idx}.npy")
                cv2.imwrite(crop_path, face_crop_bgr)
                emb = np.array(f.embedding, dtype=np.float32)
                np.save(enc_path, emb)

                # store meta
                meta = {
                    "face_idx": face_idx,
                    "frame_idx": frame_idx,
                    "frame_path" : frame_path,
                    "time_sec": frame_idx / fps,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_path": crop_path,
                    "enc_path": enc_path,
                    "quality": float(quality),
                    "det_score": float(det_score),
                }

                meta_list.append(meta)
                embeddings.append(emb)
                face_idx += 1

                if MAX_FACES and face_idx >= MAX_FACES:
                    break

            frame_idx += 1
            if pbar:
                pbar.update(1)
    finally:
        cap.release()
        if pbar:
            pbar.close()

    if len(embeddings) == 0:
        print("No faces extracted.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    embeddings = np.vstack(embeddings).astype(np.float32)
    if verbose:
        print(f"Extracted {embeddings.shape[0]} faces. Running HDBSCAN clustering...")

    # Optional: normalize embeddings for Euclidean/ cosine stability
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                min_samples=HDBSCAN_MIN_SAMPLES,
                                metric='euclidean',
                                cluster_selection_method='eom',
                                prediction_data=True)

    # labels array length == number of detections
    labels = clusterer.fit_predict(embeddings)
    for i, label in enumerate(labels):
        m = meta_list[i]
        m["cluster_id"] = int(label)
        m["hbdscan_cluster_label"] = int(label)

    if verbose:
        n_clusters = len(set([l for l in labels if l != -1]))
        n_noise = int(sum(1 for l in labels if l == -1))
        print(f"HDBSCAN produced {n_clusters} clusters, noise={n_noise}")


    # --------------------------------------------------
    # 1. Compute centroid for each cluster
    # --------------------------------------------------
    unique_labels = np.unique([l for l in labels if l != -1])
    cluster_means = []
    cluster_members = {}

    noise = embeddings[np.where(labels == -1)[0]]

    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        cluster_members[lbl] = idxs
        cluster_means.append(np.mean(embeddings[idxs], axis=0))

    cluster_means = np.array(cluster_means)

    # --------------------------------------------------
    # 2. Merge similar clusters based on cosine similarity
    # --------------------------------------------------
    sim = cosine_similarity(cluster_means)
    threshold = 0.58  # tweak 0.58–0.65 range

    G = nx.Graph()
    for i in range(len(sim)):
        for j in range(i + 1, len(sim)):
            if sim[i, j] > threshold:
                G.add_edge(i, j)

    # e.g. [{1, 2, 3}, {4, 5}, {6, 7, 8, 9, 11, 13}, {16, 17, 12}, {18, 20, 22, 15}]
    connected = list(nx.connected_components(G))
    # breakpoint()

    # now add all the ids that weren't merged together.
    clustered_labels = {element for s in connected for element in s}
    for label in unique_labels:
        label_int = int(label)
        if label_int not in clustered_labels:
            connected.append(set([label_int]))


    # Create merged cluster mapping
    merged_clusters = None
    if connected:
        merged_clusters = {}
        for new_label, comp in enumerate(connected):
            for m in meta_list:
                if m["hbdscan_cluster_label"] in comp:
                    m["cluster_id"] = new_label
            merged_clusters[new_label] = np.concatenate([cluster_members[c] for c in comp])

        print(f"Merged {len(unique_labels)} clusters → {len(merged_clusters)} clusters")

    # for analyzing two separate clusters
    # known_duplicates = {
    #     "darren": [4,5],
    # }
    #
    # for person, idxs in known_duplicates.items():
    #     print(f"\n{person}:")
    #     for i in idxs:
    #         if i == idxs[-1]:
    #             break
    #         i_embeddings = embeddings[labels == i]
    #         i_meta = [d for d in meta_list if d["hbdscan_cluster_label"] == i]
    #         for j in idxs[1:]:
    #             j_embeddings = embeddings[labels == j]
    #             j_meta = [d for d in meta_list if d["hbdscan_cluster_label"] == j]
    #             for ie, im in zip(i_embeddings, i_meta):
    #                 for je, jm in zip(j_embeddings, j_meta):
    #                     breakpoint()
    #                     dist = cosine_distances(
    #                     ie.reshape(1, -1),
    #                     je.reshape(1, -1)
    #                     )[0, 0]
    #                     print(f"  Label {i} (crop {im['crop_path']}) vs {jm['crop_path']}: distance={dist:.3f}")
    
    # breakpoint()

    # Post-process clusters
    clusters = defaultdict(list)
    for m in meta_list:
        clusters[m.get('cluster_id', -1)].append(m)


    results_dir = os.path.join(out_folder, "clusters")
    os.makedirs(results_dir, exist_ok=True)

    summary = {"clusters": {}, "params": {
        "frame_interval": FRAME_INTERVAL,
        "min_face_size": MIN_FACE_SIZE,
        "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE
    }}

    top_candidates = select_temporal_candidates(clusters, total_frames / fps)

    for label, metadata_list in top_candidates.items():
        cluster_name = f"cluster_{label:03d}"
        cluster_dir = os.path.join(results_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        for i,m in enumerate(metadata_list):

            frame = cv2.imread(m["frame_path"])
            profile_name = f"{i}_profile.jpg"
            profile_path = os.path.join(cluster_dir, profile_name)
            thumbnail_name = f"{i}_thumbnail.jpg"
            thumbnail_path = os.path.join(cluster_dir, thumbnail_name)
            # breakpoint() 
            profile_crop = crop.crop_profile_image(frame, m["bbox"])
            thumbnail_crop = crop.crop_thumbnail_16x9(frame, m["bbox"])

            cv2.imwrite(profile_path, profile_crop)
            cv2.imwrite(thumbnail_path, thumbnail_crop)

    # for label, items in clusters.items():
    #     if label == -1:
    #         # noise cluster: optionally put in separate folder
    #         lab_name = "noise"
    #     else:
    #         lab_name = f"cluster_{label:03d}"
    #
    #     lab_dir = os.path.join(results_dir, lab_name)
    #     os.makedirs(lab_dir, exist_ok=True)
    #
    #     # compute centroid embedding for non-noise
    #     emb_stack = np.vstack([np.load(item['enc_path']) for item in items])
    #     centroid = emb_stack.mean(axis=0)
    #
    #     # cluster quality: average quality and size
    #     avg_quality = float(np.mean([item['quality'] for item in items]))
    #     size = len(items)
    #
    #     frame_idxs = np.array([item['frame_idx'] for item in items])
    #     median_frame = np.median(frame_idxs)
    #
    #     best = max(items, key=lambda x:x["quality"])
    #     # copy the best crop to cluster folder
    #     rep_src = best['crop_path']
    #     rep_dst = os.path.join(lab_dir, f"representative_face_{best['face_idx']}.jpg")
    #     shutil.copy(rep_src, rep_dst)
    #
    #     # save some sample faces (up to 100) for quick review
    #     sample_items = sorted(items, key=lambda it: (-it['quality'], abs(it['frame_idx'] - median_frame)))[:100]
    #     sample_dir = os.path.join(lab_dir, "samples")
    #     os.makedirs(sample_dir, exist_ok=True)
    #     for s in sample_items:
    #         shutil.copy(s['crop_path'], os.path.join(sample_dir, os.path.basename(s['crop_path'])))
    #
    #     # write cluster metadata
    #     summary['clusters'][str(label)] = {
    #         "label": int(label),
    #         "size": int(size),
    #         "avg_quality": float(avg_quality),
    #         "representative": os.path.relpath(rep_dst, out_folder),
    #         "samples_dir": os.path.relpath(sample_dir, out_folder),
    #         "centroid_norm": float(np.linalg.norm(centroid)),
    #     }

    # save meta list and summary
    with open(os.path.join(out_folder, "detections.json"), "w") as f:
        json.dump(meta_list, f, indent=2)
    # with open(os.path.join(out_folder, "summary.json"), "w") as f:
    #     json.dump(summary, f, indent=2)

    # cleanup optional: keep tmp for debugging if verbose, else remove
    if verbose:
        print(f"Clusters saved to {results_dir}")
        print(f"Summary metadata saved to {os.path.join(out_folder, 'summary.json')}")
        print("Temporary files left in:", tmp_dir)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster faces in a video using InsightFace + HDBSCAN")
    parser.add_argument("video", help="input video file (mp4)")
    parser.add_argument("out", help="output folder")
    parser.add_argument("--gpu", action="store_true", help="use GPU if available")
    parser.add_argument("--resize", type=int, default=None, help="resize frame width for speed")
    parser.add_argument("--frame-interval", type=int, default=FRAME_INTERVAL, help="process every Nth frame")
    args = parser.parse_args()

    if args.resize:
        RESIZE_WIDTH = args.resize
    FRAME_INTERVAL = args.frame_interval

    pipeline(args.video, args.out, use_gpu=args.gpu, verbose=True)
