import os
import pickle

import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm

class FaceDetection:
    def __init__(self, model_name="buffalo_l", frame_interval=30, min_face_size=80, verbose=True):
        self.model_name = model_name
        self.detector = FaceAnalysis(
            name=model_name, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

        self.frame_interval = frame_interval
        self.min_face_size = min_face_size
        self.verbose = verbose
    
    def detect_faces_video(self, video_path: str, output_dir_path: str):
        """
        Outputs face crops and pickle file including encodings and 
        metadata to output_dir_path
        """
        os.makedirs(output_dir_path, exist_ok=True)
        crops_dir = os.path.join(output_dir_path, "crops")
        enc_dir = os.path.join(output_dir_path, "face_data")
        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(enc_dir, exist_ok=True)

        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Unable to open video: " + video_path)


        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.verbose:
            print(f"Video opened: {video_path}, frames={total_frames}, fps={fps}")

        frame_idx = 0
        face_idx = 0

        pbar = tqdm(total=total_frames // self.frame_interval + 1, desc="Frames") if self.verbose else None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_interval != 0:
                    frame_idx += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # returns list of Face objects with bbox, kps, embedding, det_score, pose, etc.
                faces = self.detector.get(rgb)  


                for f in faces:
                    if f["det_score"] < 0.7:
                        continue


                    x1, y1, x2, y2 = map(int, f.bbox.round().astype(int))
                    margin = 0.5
                    w = f.bbox[2] - f.bbox[0]
                    h = f.bbox[3] - f.bbox[1]
                    if w < self.min_face_size or h < self.min_face_size:
                        continue
                    
                    x1 = max(0, int(f.bbox[0] - w * margin))
                    y1 = max(0, int(f.bbox[1] - h * margin))
                    x2 = min(frame.shape[1], int(f.bbox[2] + w * margin))
                    y2 = min(frame.shape[0], int(f.bbox[3] + h * margin))
                    
                    # crop BGR face (for quality scoring & saving)
                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size == 0:
                        continue

                    crop_path = os.path.join(crops_dir, f"face_{face_idx}.jpg")
                    cv2.imwrite(crop_path, crop)

                    face_pickle_path = os.path.join(enc_dir, f"face_{face_idx}.pkl")
                    with open(face_pickle_path, 'wb') as pkl_file:
                        pickle.dump(dict(f), pkl_file)

                    face_idx += 1

                frame_idx += 1
                if pbar:
                    pbar.update(1)
        finally:
            cap.release()
            if pbar:
                pbar.close()

if __name__ == "__main__":
    video_path = "./data/videos/slow_jerk.mp4"
    output_path = "./data/faces/slow_jerk"

    detector = FaceDetection()
    detector.detect_faces_video(video_path, output_path)

    crops = os.listdir(os.path.join(output_path, "crops"))
    face_data = os.listdir(os.path.join(output_path, "face_data"))

    print(f"Outputted {len(crops)} crops and {len(face_data)} face data pkl files")
