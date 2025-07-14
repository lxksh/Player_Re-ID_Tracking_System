import cv2
import json
import numpy as np
import os
import sys


def evaluate_tracking_output(video_path, metrics_path):
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return []

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    print("TRACKING EVALUATION REPORT")
    print("=" * 50)

    print("\nAvailable metrics:")
    for key in metrics.keys():
        if not isinstance(metrics[key], (list, dict)):
            print(f"  - {key}: {metrics[key]}")

    quality_issues = []

    total_tracks = metrics.get("total_tracks", 0)
    if total_tracks > 22:
        quality_issues.append(
            f"Too many tracks detected: {total_tracks}. Possible ID mismatches."
        )
    elif total_tracks == 0:
        quality_issues.append(
            "No tracks found. Check your detection and tracking setup."
        )

    avg_fps = metrics.get("avg_fps", None)
    if avg_fps is not None and avg_fps < 15:
        quality_issues.append(
            f"Low processing speed: average FPS is {avg_fps:.1f}. Consider optimization."
        )
    elif avg_fps is None:
        if "processing_times" in metrics and metrics["processing_times"]:
            avg_time = np.mean(metrics["processing_times"])
            calculated_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\nCalculated FPS based on processing times: {calculated_fps:.1f}")

    if "track_lifetimes" in metrics and metrics["track_lifetimes"]:
        lifetimes = [int(v) for v in metrics["track_lifetimes"].values()]
        avg_lifetime = sum(lifetimes) / len(lifetimes)

        if "video_info" in metrics:
            fps = metrics["video_info"].get("fps", 30)
            avg_lifetime_seconds = avg_lifetime / fps
            print(
                f"\nAverage track lifetime: {avg_lifetime:.1f} frames ({avg_lifetime_seconds:.1f} seconds)"
            )

            if avg_lifetime < fps:
                quality_issues.append(
                    f"Short track lifetimes detected: {avg_lifetime:.1f} frames on average."
                )

    if "detection_counts" in metrics and len(metrics["detection_counts"]) > 0:
        det_counts = metrics["detection_counts"]
        det_variance = np.var(det_counts)
        det_mean = np.mean(det_counts)
        det_std = np.std(det_counts)

        print("\nDetection statistics:")
        print(f"  - Mean detections per frame: {det_mean:.1f}")
        print(f"  - Standard deviation: {det_std:.1f}")
        print(f"  - Range: {min(det_counts)} to {max(det_counts)}")

        if det_variance > 10:
            quality_issues.append(
                "Detection counts vary widely. Detector may need fine-tuning."
            )

    if "frame_count" in metrics and "video_info" in metrics:
        processed = metrics["frame_count"]
        total = metrics["video_info"].get("total_frames", processed)
        if processed < total * 0.95:
            quality_issues.append(
                f"Processing incomplete: {processed} out of {total} frames processed."
            )

    print("\n" + "-" * 50)
    if quality_issues:
        print("Potential issues found:")
        for i, issue in enumerate(quality_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No major quality issues detected.")

    if "statistics" in metrics:
        print("\nAdditional statistics:")
        stats = metrics["statistics"]
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")

    print("\n" + "-" * 50)
    print("Visual inspection checklist:")
    print("[ ] Player IDs remain consistent when players are visible")
    print("[ ] Players leaving and re-entering keep the same ID")
    print("[ ] No ID switches between players")
    print("[ ] Bounding boxes are stable and accurate")
    print("[ ] Trajectories look smooth")
    print("[ ] All players are detected reliably")

    print("\n" + "-" * 50)
    print("Recommendations:")

    if total_tracks > 22:
        print("- Adjust minimum hits to reduce false positives")
        print("- Refine feature similarity thresholds for better matching")

    if avg_fps and avg_fps < 20:
        print("- Consider lowering detection confidence threshold")
        print("- Use smaller feature vectors for faster computation")

    if "track_lifetimes" in metrics:
        lifetimes = [int(v) for v in metrics["track_lifetimes"].values()]
        avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0
        if avg_lifetime < 30:
            print("- Extend maximum track age to keep tracks alive longer")
            print("- Improve feature extraction to help re-identify players")

    return quality_issues


def check_video_properties(video_path):
    print("\n" + "-" * 50)
    print("Video properties:")

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  - Resolution: {width} x {height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {frame_count}")
        print(f"  - Duration: {frame_count / fps:.1f} seconds")

        cap.release()
    else:
        print("  - Unable to open the video file")


def analyze_tracking_performance(metrics_path):
    print("\n" + "-" * 50)
    print("Performance analysis:")

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if "track_lifetimes" in metrics:
            lifetimes = list(metrics["track_lifetimes"].values())

            long_tracks = sum(1 for l in lifetimes if l > 60)
            short_tracks = sum(1 for l in lifetimes if l < 30)

            print("\nTrack duration breakdown:")
            print(f"  - Long tracks (>2 seconds): {long_tracks}")
            print(f"  - Short tracks (<1 second): {short_tracks}")
            print(f"  - Total tracks: {len(lifetimes)}")

            if len(lifetimes) > 0:
                fragmentation_rate = short_tracks / len(lifetimes)
                print(f"  - Fragmentation rate: {fragmentation_rate:.2%}")

                if fragmentation_rate > 0.5:
                    print("  High fragmentation detected. Many tracks are too short.")
                else:
                    print("  Track durations look acceptable.")

        if "processing_times" in metrics and len(metrics["processing_times"]) > 0:
            times_ms = [t * 1000 for t in metrics["processing_times"]]
            print("\nProcessing time per frame:")
            print(f"  - Average: {np.mean(times_ms):.1f} ms")
            print(f"  - Minimum: {np.min(times_ms):.1f} ms")
            print(f"  - Maximum: {np.max(times_ms):.1f} ms")
            print(f"  - Standard deviation: {np.std(times_ms):.1f} ms")

    except Exception as e:
        print(f"  Error reading performance data: {e}")


def generate_improvement_script(quality_issues):
    print("\n" + "-" * 50)
    print("Suggested parameter changes:")

    if any("Too many tracks" in issue for issue in quality_issues):
        print("\nExample adjustments to reduce ID mismatches:")
        print("tracker = PlayerTracker(max_age=20, min_hits=5)")
        print("detector = PlayerDetector(model_path, conf_threshold=0.6)")

    if any("Short track lifetimes" in issue for issue in quality_issues):
        print("\nExample adjustments to improve track continuity:")
        print("tracker = PlayerTracker(max_age=45, min_hits=2)")

    if any("Low processing speed" in issue for issue in quality_issues):
        print("\nTo improve performance, you could try:")
        print("- Reducing feature vector size")
        print("- Skipping alternate frames")
        print("- Using a faster feature extractor")


if __name__ == "__main__":
    video_path = "data/output/tracked_output.mp4"
    metrics_path = "data/output/tracking_metrics.json"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        metrics_path = sys.argv[2]

    print(f"Evaluating tracking output for: {video_path}")
    print(f"Using metrics file: {metrics_path}")

    issues = evaluate_tracking_output(video_path, metrics_path)
    check_video_properties(video_path)
    analyze_tracking_performance(metrics_path)
    generate_improvement_script(issues)

    print("\n" + "-" * 50)
    print("Raw metrics structure:")
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            for key in sorted(metrics.keys()):
                value_type = type(metrics[key]).__name__
                if isinstance(metrics[key], list):
                    print(f"  - {key}: {value_type} (length: {len(metrics[key])})")
                elif isinstance(metrics[key], dict):
                    print(f"  - {key}: {value_type} (keys: {len(metrics[key])})")
                else:
                    print(f"  - {key}: {value_type}")
    except Exception as e:
        print(f"Error reading metrics: {e}")

    print("\n" + "=" * 50)
    print("Evaluation completed.")
