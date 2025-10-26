import os
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional
import tempfile
import shutil

import cv2
import hdbscan
from insightface.app import FaceAnalysis
import networkx as nx
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

import crop


class InvalidInputError(Exception):
    """Custom exception for invalid input."""
    pass


@dataclass
class FaceMeta:
    bbox: list[int]
    quality: float
    det_score: float

    face_idx: Optional[int] = None
    frame_idx: Optional[int] = None
    frame_path: Optional[str] = None
    time_sec: Optional[float] = None
    crop_path: Optional[str] = None
    enc_path: Optional[str] = None
    cluster_id: Optional[int] = None
    hbdscan_label: Optional[int] = None


class Screenshot:
    def __init__(
        self, 
        quality_model_path,
        resize_width=224,
        frame_interval=None, # if None, dynamic based on fps but can less or more for speed vs quality
        detection_threshold=0.6,
        image_quality_threshold=1.1,
        cosine_similarity_threshold=0.58,
        hdbscan_min_cluster_size=3,
        hdbscan_min_samples=2,
        verbose=False,
        debug=False
    ):
        if not quality_model_path or not os.path.exists(quality_model_path):
            raise InvalidInputError("Cast image quality model does not exist")

        # init cast image quality model inference
        base = models.resnet50(pretrained=False)
        base.fc = nn.Linear(base.fc.in_features, 1)
        model = base
        model.load_state_dict(torch.load(quality_model_path, map_location="cuda"))
        model.eval().cuda()

        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


        # InsightFace Face Detection
        app = FaceAnalysis(
            name="buffalo_l", 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 0 means GPU
        app.prepare(ctx_id=0, det_size=(640, 640))
        self.app = app


        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                    min_samples=hdbscan_min_samples,
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    prediction_data=True)
        self.frame_interval = frame_interval
        self.resize_width = resize_width
        self.detection_threshold = detection_threshold
        self.image_quality_threshold = image_quality_threshold
        self.cosine_threshold = cosine_similarity_threshold
        self.hdbscan_min_cluster = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.verbose = verbose
        if debug:
            self.debug = debug
            self.verbose = True

    @torch.no_grad()
    def predict_quality(self, face_resized):
        # Convert NumPy array (BGR from cv2) to PIL Image (RGB)
        img = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0).cuda()  # shape: (1, 3, 224, 224)
        score = self.model(img)
        return score.item()

    def determine_min_face_size(self, frame_h):
        ref_height = 720
        ref_face_size = 45

        # Linear scaling factor
        scale = frame_h / ref_height

        # Scale face size linearly with height
        min_face_size = int(ref_face_size * scale)

        # Clamp to reasonable range (so very small or very large videos don't break)
        min_face_size = max(25, min(min_face_size, 100))

        return min_face_size
    def select_temporal_candidates(self, clusters: defaultdict[int, dict], video_duration_sec, num_candidates=3) -> defaultdict[int, list]:
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
                cluster_candidates[label] = sorted(metadata, key=lambda m: m.time_sec)
                continue

            # Divide video into equal segments (this could implement this by getting the span
            # a person(cluster) appears in a video and find candidates in this span instead)
            segment_duration = video_duration_sec / num_candidates

            for segment_idx in range(num_candidates):
                segment_start = segment_idx * segment_duration
                segment_end = (segment_idx + 1) * segment_duration

                # Find meta_lists within this segment
                segment_meta_list = [
                    m for m in metadata
                    if segment_start <= m.time_sec < segment_end
                ]

                if segment_meta_list:
                    # Pick the highest quality item from this segment
                    best = max(segment_meta_list, key=lambda m: m.quality)
                    cluster_candidates[label].append(best)

        return cluster_candidates

    def detect_faces(self, frame, min_face_size) -> tuple[list[FaceMeta], list]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        meta_list, emb_list = [], []
        for f in faces:
            det_score = float(f.det_score) if hasattr(f, "det_score") else 1.0
            if det_score < self.detection_threshold:
                continue

            # f.bbox: [x1,y1,x2,y2], f.kps: landmarks, f.det_score, f.embedding (ArcFace)
            x1, y1, x2, y2 = map(int, f.bbox.round().astype(int))
            # ensure box inside frame
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
            w = x2 - x1; h = y2 - y1
            if w < min_face_size or h < min_face_size:
                continue

            # crop BGR face (for quality scoring & saving)
            face_crop_bgr = crop.crop_face_square(frame, f.bbox, self.resize_width)
            if face_crop_bgr is None or face_crop_bgr.size == 0:
                continue

            quality = self.predict_quality(face_crop_bgr)
            if quality < self.image_quality_threshold:
                continue

            meta = FaceMeta(
                bbox=[int(x1), int(y1), int(x2), int(y2)],
                quality=float(quality),
                det_score=float(det_score),
            )
            emb = np.array(f.embedding, dtype=np.float32)

            meta_list.append(meta)
            emb_list.append(emb)

        return meta_list, emb_list

    def cluster(self, meta_list, embeddings) -> defaultdict:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = self.clusterer.fit_predict(embeddings)
        for i, label in enumerate(labels):
            m = meta_list[i]
            m.cluster_id = int(label)
            m.hbdscan_label = int(label)

        if self.verbose:
            n_clusters = len(set([l for l in labels if l != -1]))
            n_noise = int(sum(1 for l in labels if l == -1))
            print(f"HDBSCAN produced {n_clusters} clusters, noise={n_noise}")

        # 1. Compute centroid for each cluster
        unique_labels = np.unique([l for l in labels if l != -1])
        cluster_means = []
        cluster_members = {}

        # noise = embeddings[np.where(labels == -1)[0]]

        for lbl in unique_labels:
            idxs = np.where(labels == lbl)[0]
            cluster_members[int(lbl)] = idxs
            cluster_means.append(np.mean(embeddings[idxs], axis=0))

        cluster_means = np.array(cluster_means)

        # 2. Merge similar clusters based on cosine similarity
        sim = cosine_similarity(cluster_means)

        G = nx.Graph()
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                if sim[i, j] > self.cosine_threshold:
                    G.add_edge(i, j)

        # e.g. [{1, 2, 3}, {4, 5}, {6, 7, 8, 9, 11, 13}, {16, 17, 12}, {18, 20, 22, 15}]
        connected = list(nx.connected_components(G))

        # now add all the ids that weren't merged together.
        clustered_labels = {element for s in connected for element in s}
        for label in unique_labels:
            label_int = int(label)
            if label_int not in clustered_labels:
                connected.append(set([label_int]))

        # Create merged cluster mapping
        merged_clusters = {}
        for new_label, comp in enumerate(connected):
            merged_clusters[new_label] = np.concatenate([cluster_members[c] for c in comp])
        if self.verbose:
            print(f"Cosine similarity merged {len(unique_labels)} clusters → {len(merged_clusters)} clusters")

        clusters = defaultdict(list)
        for new_label, idxs in merged_clusters.items():
            for i in idxs:
                m = meta_list[i]
                m.cluster_id = new_label
                clusters[m.cluster_id].append(m)

        return clusters


    def screenshot(self, video_path, out_dir, max_faces=None):
        cap = None
        frame_idx = 0
        face_idx = 0
        meta_list = []
        embeddings = []
        os.makedirs(out_dir, exist_ok=True)

        tmp_dir = tempfile.mkdtemp(prefix="face_cluster_")
        crops_dir = os.path.join(tmp_dir, "crops")
        enc_dir = os.path.join(tmp_dir, "encs")
        frames_dir = os.path.join(tmp_dir, "frames")

        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(enc_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open video: " + video_path)

        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = self.frame_interval if self.frame_interval else int(fps)

        min_face_size = self.determine_min_face_size(frame_height)

        if self.verbose:
            print(f"Video opened: {video_path}, dimensions: {frame_width}x{frame_height} frames={total_frames}, fps={fps}")
            print(f"Minimum face size: {min_face_size}")

        pbar = tqdm(total=total_frames // frame_interval + 1, desc="Frames") if self.verbose else None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                frame = crop.resize_for_detection(frame)
                frame = crop.remove_borders(frame)

                frame_name = f"frame_{frame_idx:04d}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                cv2.imwrite(frame_path, frame)

                frame_metas, frame_embs = self.detect_faces(frame, min_face_size)
                for m, emb in zip(frame_metas, frame_embs):
                    crop_path = os.path.join(crops_dir, f"face_{face_idx}.jpg")
                    enc_path = os.path.join(enc_dir, f"enc_{face_idx}.npy")

                    face_crop_bgr = crop.crop_face_square(frame, m.bbox, self.resize_width)
                    if face_crop_bgr is None:
                        continue

                    m.face_idx = face_idx
                    m.frame_idx = frame_idx
                    m.frame_path = frame_path
                    m.time_sec = frame_idx / fps
                    m.crop_path = crop_path
                    m.enc_path = enc_path

                    cv2.imwrite(crop_path, face_crop_bgr)
                    np.save(enc_path, emb)
                    meta_list.append(m)
                    embeddings.append(emb)

                    face_idx += 1

                if max_faces and face_idx >= max_faces:
                    print("Max faces detected!")
                    break

                frame_idx += 1
                if pbar:
                    pbar.update(1)
                    pbar.refresh()
        finally:
            if cap:
                cap.release()
            if pbar:
                pbar.close()

        if len(embeddings) == 0:
            print("No faces extracted.")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

        embeddings = np.vstack(embeddings).astype(np.float32)
        if self.verbose:
            print(f"Extracted {embeddings.shape[0]} faces. Running HDBSCAN clustering...")

        # normalize embeddings for Euclidean/ cosine stability
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # returns MetaFace objects grouped by a cluster id and
        # adds cluster labels to each MetaFace object
        clusters = self.cluster(meta_list, embeddings)

        top_candidates = self.select_temporal_candidates(clusters, total_frames / fps)
        # output the top candiates crops per cluster
        for label, metadata_list in top_candidates.items():
            for i,m in enumerate(metadata_list):
                frame = cv2.imread(m.frame_path)

                profile_name = f"{label:03d}_cluster_{i}_profile.jpg"
                profile_path = os.path.join(out_dir, profile_name)
                profile_crop = crop.crop_profile_image(frame, m.bbox)
                if profile_crop is not None:
                    cv2.imwrite(profile_path, profile_crop)

                thumbnail_name = f"{label:03d}_cluster_{i}_thumbnail.jpg"
                thumbnail_path = os.path.join(out_dir, thumbnail_name)
                thumbnail_crop = crop.crop_thumbnail_16x9(frame, m.bbox)
                if thumbnail_crop is not None:
                    cv2.imwrite(thumbnail_path, thumbnail_crop)

        if self.verbose:
            print(f"Clusters candidate shots saved to {out_dir}")

        if self.debug:
            print("Temporary files left in:", tmp_dir)
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Screenshot people in a video"
    )

    parser.add_argument(
        "-i", "--input",
        help="path to video file",
        required=True
    )
    parser.add_argument(
        "-m", "--model", 
        help="path to image quality model", 
        required=True
    )
    parser.add_argument(
        "-o", "--output", 
        help="path to directory where cluster images are saved", 
        required=True
    )
    args = parser.parse_args()
    if not os.path.exists(args.model):
        print("Model does not exist")
        sys.exit(1)

    sc = Screenshot(
        quality_model_path=args.model,
        debug=True
    )

    sc.screenshot(args.input, args.output)

