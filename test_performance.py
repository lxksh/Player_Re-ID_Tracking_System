import time
import numpy as np
from src.feature_extractor import FeatureExtractor

fe = FeatureExtractor()
dummy_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
dummy_bboxes = np.random.rand(20, 4) * [1280, 720, 1280, 720]

start = time.time()
for _ in range(10):
    features = fe.extract(dummy_image, dummy_bboxes)
elapsed = time.time() - start

print(f"Feature extraction: {elapsed / 10 * 1000:.1f} ms per frame")
print(f"Expected FPS with optimization: {1 / (elapsed / 10):.1f}")
