
import cv2
import numpy as np
import json
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_video_info(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        / int(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return info


def calculate_center(bbox: np.ndarray) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def calculate_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def normalize_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height])


def denormalize_bbox(bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [
            int(x1 * img_width),
            int(y1 * img_height),
            int(x2 * img_width),
            int(y2 * img_height),
        ]
    )


def save_tracking_results(tracks: Dict, output_path: str):
    results = {"tracks": []}

    for track_id, track_data in tracks.items():
        track_info = {
            "id": track_id,
            "frames": track_data["frames"],
            "bboxes": track_data["bboxes"].tolist()
            if isinstance(track_data["bboxes"], np.ndarray)
            else track_data["bboxes"],
            "lifetime": len(track_data["frames"]),
        }
        results["tracks"].append(track_info)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Tracking results saved to {output_path}")


def interpolate_bbox(bbox1: np.ndarray, bbox2: np.ndarray, alpha: float) -> np.ndarray:
    return bbox1 * (1 - alpha) + bbox2 * alpha


def smooth_trajectory(
    points: List[Tuple[int, int]], window_size: int = 5
) -> List[Tuple[int, int]]:
    if len(points) < window_size:
        return points

    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)

        window_points = points[start:end]
        avg_x = sum(p[0] for p in window_points) / len(window_points)
        avg_y = sum(p[1] for p in window_points) / len(window_points)

        smoothed.append((int(avg_x), int(avg_y)))

    return smoothed


class Timer:

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = cv2.getTickCount()
        return self

    def __exit__(self, *args):
        end_time = cv2.getTickCount()
        time_ms = (end_time - self.start_time) / cv2.getTickFrequency() * 1000
        logger.debug(f"{self.name} took {time_ms:.2f} ms")
