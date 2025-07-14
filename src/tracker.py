import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, track_id, bbox, feature):
        self.id = track_id
        self.bbox = bbox
        self.feature = feature
        self.age = 0
        self.hits = 1
        self.time_since_update = 0


class PlayerTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.lost_tracks = {}

    def update(self, bboxes, features):
        
        for track in self.tracks.values():
            track.time_since_update += 1
            track.age += 1

        if len(bboxes) == 0:
            return self._get_active_tracks()

        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            bboxes, features
        )

        for det_idx, track_id in matched:
            self.tracks[track_id].bbox = bboxes[det_idx]
            self.tracks[track_id].feature = features[det_idx]
            self.tracks[track_id].time_since_update = 0
            self.tracks[track_id].hits += 1

        for det_idx in unmatched_dets:
            best_match = None
            best_sim = 0.5

            for lost_id, lost_track in self.lost_tracks.items():
                sim = self._cosine_similarity(features[det_idx], lost_track.feature)
                if sim > best_sim:
                    best_sim = sim
                    best_match = lost_id

            if best_match:
                self.tracks[best_match] = self.lost_tracks[best_match]
                self.tracks[best_match].bbox = bboxes[det_idx]
                self.tracks[best_match].feature = features[det_idx]
                self.tracks[best_match].time_since_update = 0
                del self.lost_tracks[best_match]
            else:
                self.tracks[self.next_id] = Track(
                    self.next_id, bboxes[det_idx], features[det_idx]
                )
                self.next_id += 1

        for track_id in unmatched_tracks:
            if self.tracks[track_id].time_since_update > self.max_age:
                self.lost_tracks[track_id] = self.tracks[track_id]
                del self.tracks[track_id]

        return self._get_active_tracks()

    def _match_detections(self, bboxes, features):
        if not self.tracks:
            return [], list(range(len(bboxes))), []

        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(bboxes), len(track_ids)))

        for i, (bbox, feat) in enumerate(zip(bboxes, features)):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]

                iou = self._compute_iou(bbox, track.bbox)
                iou_cost = 1 - iou

                feat_sim = self._cosine_similarity(feat, track.feature)
                feat_cost = 1 - feat_sim

                cost_matrix[i, j] = 0.5 * iou_cost + 0.5 * feat_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0.7:
                matched.append((r, track_ids[c]))

        unmatched_dets = [
            i for i in range(len(bboxes)) if i not in [m[0] for m in matched]
        ]
        unmatched_tracks = [t for t in track_ids if t not in [m[1] for m in matched]]

        return matched, unmatched_dets, unmatched_tracks

    def _compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _cosine_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6
        )

    def _get_active_tracks(self):
        return [track for track in self.tracks.values() if track.hits >= self.min_hits]
