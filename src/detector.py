import numpy as np
from ultralytics import YOLO
import torch


class PlayerDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Detector using device: {self.device}")

    def detect(self, frame):
        results = self.model(
            frame, conf=self.conf_threshold, device=self.device, verbose=False
        )

        bboxes = []
        scores = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    score = box.conf[0].cpu().numpy()
                    bboxes.append(bbox)
                    scores.append(score)

        return np.array(bboxes), np.array(scores)
