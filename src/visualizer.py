import cv2
import numpy as np
from collections import defaultdict


class Visualizer:
    def __init__(self, track_length=30):
        self.track_length = track_length
        self.tracks_history = defaultdict(list)
        self.colors = self._generate_colors(100)

    def _generate_colors(self, n):
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors

    def draw_tracks(self, frame, tracks):
        for track in tracks:
            track_id = track.id
            bbox = track.bbox.astype(int)
            color = self.colors[track_id % len(self.colors)]

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            label = f"ID: {track_id}"
            cv2.putText(
                frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            self.tracks_history[track_id].append(center)

            if len(self.tracks_history[track_id]) > self.track_length:
                self.tracks_history[track_id].pop(0)

            points = np.array(self.tracks_history[track_id], dtype=np.int32)
            if len(points) > 1:
                cv2.polylines(frame, [points], False, color, 2)

        return frame
