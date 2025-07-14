#!/usr/bin/env python3

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import argparse
from collections import defaultdict
import torch

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.feature_extractor import FeatureExtractor

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


class TrackingVisualizer:
    def __init__(self, track_length=30):
        self.track_length = track_length
        self.tracks_history = defaultdict(list)
        self.colors = self._generate_colors(50)

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
                0.5,
                color,
                2,
            )

            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            self.tracks_history[track_id].append(center)

            if len(self.tracks_history[track_id]) > self.track_length:
                self.tracks_history[track_id].pop(0)

            points = self.tracks_history[track_id]
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i - 1], points[i], color, 1)

        return frame


def process_video(video_path, model_path, output_path, display=False):
    print("Initializing components...")
    detector = PlayerDetector(model_path, conf_threshold=0.5)
    tracker = PlayerTracker(max_age=30, min_hits=3)
    feature_extractor = FeatureExtractor()
    visualizer = TrackingVisualizer(track_length=30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} at {fps} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    metrics = {
        "frame_count": 0,
        "total_tracks": 0,
        "processing_times": [],
        "active_tracks_per_frame": [],
        "track_lifetimes": {},
        "detection_counts": [],
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        },
    }

    track_ids_seen = set()
    frame_first_seen = {}
    frame_last_seen = {}

    print(f"\nProcessing {total_frames} frames")
    pbar = tqdm(total=total_frames, desc="Tracking")

    frame_idx = 0
    frame_skip = 1

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            start_time = time.time()

            bboxes, scores = detector.detect(frame)
            metrics["detection_counts"].append(len(bboxes))

            features = feature_extractor.extract(frame, bboxes)

            active_tracks = tracker.update(bboxes, features)

            for track in active_tracks:
                track_id = track.id
                track_ids_seen.add(track_id)

                if track_id not in frame_first_seen:
                    frame_first_seen[track_id] = frame_idx
                frame_last_seen[track_id] = frame_idx

            vis_frame = visualizer.draw_tracks(frame.copy(), active_tracks)

            info_text = f"Frame: {frame_idx} | Active: {len(active_tracks)} | Total: {len(track_ids_seen)}"
            cv2.putText(
                vis_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            frame_time = time.time() - start_time
            metrics["processing_times"].append(frame_time)
            metrics["active_tracks_per_frame"].append(len(active_tracks))
            metrics["frame_count"] = frame_idx + 1

            out.write(vis_frame)

            if display:
                cv2.imshow("Player Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user")
                    break

            pbar.update(1)
            frame_idx += 1

    except Exception as e:
        print(f"\nProcessing error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    for track_id in track_ids_seen:
        if track_id in frame_first_seen and track_id in frame_last_seen:
            lifetime = frame_last_seen[track_id] - frame_first_seen[track_id] + 1
            metrics["track_lifetimes"][str(track_id)] = lifetime

    metrics["total_tracks"] = len(track_ids_seen)

    if metrics["processing_times"]:
        metrics["avg_processing_time"] = np.mean(metrics["processing_times"])
        metrics["avg_fps"] = 1.0 / metrics["avg_processing_time"]
    else:
        metrics["avg_processing_time"] = 0
        metrics["avg_fps"] = 0

    metrics["statistics"] = {
        "avg_detections_per_frame": np.mean(metrics["detection_counts"])
        if metrics["detection_counts"]
        else 0,
        "max_simultaneous_tracks": max(metrics["active_tracks_per_frame"])
        if metrics["active_tracks_per_frame"]
        else 0,
        "avg_track_lifetime": np.mean(list(metrics["track_lifetimes"].values()))
        if metrics["track_lifetimes"]
        else 0,
    }

    return metrics


def save_report(metrics, output_dir):
    report_path = Path(output_dir) / "tracking_report.txt"

    with open(report_path, "w") as f:
        f.write("PLAYER RE-IDENTIFICATION TRACKING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("Tracking Results:\n")
        f.write(f"  Total Unique Players Tracked: {metrics['total_tracks']}\n")
        f.write(f"  Average Processing FPS: {metrics['avg_fps']:.2f}\n")

    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Player Re-ID Tracker")
    parser.add_argument(
        "--video", default="data/input/15sec_input_720.mp4", help="Input video path"
    )
    parser.add_argument("--model", default="data/input/best.pt", help="YOLO model path")
    parser.add_argument(
        "--output", default="data/output/tracked_output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--display", action="store_true", help="Display tracking results"
    )

    args = parser.parse_args()

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return

    try:
        metrics = process_video(args.video, args.model, args.output, args.display)

        metrics_path = output_dir / "tracking_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        save_report(metrics, output_dir)

        print("\n" + "=" * 50)
        print("Tracking complete")
        print("=" * 50)
        print(f"Output video: {args.output}")
        print(f"Total unique players tracked: {metrics['total_tracks']}")
        print(f"Average processing FPS: {metrics['avg_fps']:.2f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
