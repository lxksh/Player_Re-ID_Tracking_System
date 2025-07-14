import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent dir to path so `src` can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.tracker import PlayerTracker
from src.feature_extractor import FeatureExtractor
from src.utils import bbox_iou, calculate_center


# Shared fixture: sample detections for everyone
@pytest.fixture
def sample_detections():
    return np.array(
        [
            [100, 100, 150, 200],
            [200, 100, 250, 200],
            [300, 100, 350, 200],
        ]
    )


class TestTracker:
    @pytest.fixture
    def tracker(self):
        return PlayerTracker(max_age=5, min_hits=2)

    @pytest.fixture
    def sample_features(self):
        return np.random.rand(3, 128)

    def test_tracker_initialization(self, tracker):
        assert tracker.next_id == 1
        assert len(tracker.tracks) == 0
        assert tracker.max_age == 5
        assert tracker.min_hits == 2

    def test_first_frame_tracking(self, tracker, sample_detections, sample_features):
        tracks = tracker.update(sample_detections, sample_features)

        assert len(tracks) == 0
        assert len(tracker.tracks) == 3

    def test_track_confirmation(self, tracker, sample_detections, sample_features):
        tracker.update(sample_detections, sample_features)
        tracks = tracker.update(sample_detections, sample_features)

        assert len(tracks) == 3
        for track in tracks:
            assert track.hits >= 2

    def test_track_lost_and_recovery(self, tracker, sample_detections, sample_features):
        tracker.update(sample_detections, sample_features)
        tracker.update(sample_detections, sample_features)

        reduced_detections = sample_detections[:2]
        reduced_features = sample_features[:2]

        for _ in range(6):
            tracker.update(reduced_detections, reduced_features)

        assert len(tracker.lost_tracks) > 0

        tracks = tracker.update(sample_detections, sample_features)

        track_ids = [t.id for t in tracks]
        assert len(set(track_ids)) == 3


class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_feature_extraction(self, extractor, sample_image, sample_detections):
        features = extractor.extract(sample_image, sample_detections)

        assert features.shape[0] == len(sample_detections)
        assert features.shape[1] == extractor.feature_dim

        for feat in features:
            norm = np.linalg.norm(feat)
            assert abs(norm - 1.0) < 0.01  # norm should be ~1.0

    def test_empty_detections(self, extractor, sample_image):
        empty_detections = np.array([])
        features = extractor.extract(sample_image, empty_detections)

        assert len(features) == 0


class TestUtils:
    def test_bbox_iou(self):
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 150, 150])

        iou = bbox_iou(bbox1, bbox2)
        expected_iou = (50 * 50) / (100 * 100 + 100 * 100 - 50 * 50)

        assert abs(iou - expected_iou) < 0.001

    def test_calculate_center(self):
        bbox = np.array([100, 200, 300, 400])
        cx, cy = calculate_center(bbox)

        assert cx == 200
        assert cy == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
