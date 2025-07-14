import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        self.feature_dim = 128

    def extract(self, image, bboxes):
        if len(bboxes) == 0:
            return np.array([])

        features = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(self.feature_dim))
                continue

            crop = image[y1:y2, x1:x2]
            crop_small = cv2.resize(crop, (16, 32), interpolation=cv2.INTER_LINEAR)

            hist = cv2.calcHist(
                [crop_small], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]
            )
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)

            if len(hist) >= self.feature_dim:
                feature_vector = hist[: self.feature_dim]
            else:
                feature_vector = np.pad(hist, (0, self.feature_dim - len(hist)))

            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm

            features.append(feature_vector)

        return np.array(features)
