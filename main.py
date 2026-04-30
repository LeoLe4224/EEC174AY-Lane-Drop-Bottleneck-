import os
import cv2
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "out_videos"

MODEL_WEIGHTS = "runs/detect/runs_highway/car_detector_ft/weights/best.pt"
# custom model has one class only: car
VEHICLE_CLASSES = [0]

CONF_THRESH = 0.03
IMG_SIZE = 1920
LINE_WIDTH = 2

SHOW_VIDEO = False

SCREEN_WIDTH_FEET = 500.0
SPEED_WINDOW_FRAMES = 8
MIN_SPEED_SAMPLE_FRAMES = 3
MIN_HORIZONTAL_TRAVEL_PIXELS = 5.0
SPEED_CHANGE_THRESHOLD_MPH = 0.5
REPORT_FLUSH_EVERY_FRAMES = 30

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")


# =========================================================
# HELPERS
# =========================================================

def collect_videos(folder):
    vids = []

    if not os.path.isdir(folder):
        return vids

    for f in os.listdir(folder):
        fullpath = os.path.join(folder, f)

        if os.path.isfile(fullpath) and fullpath.endswith(VIDEO_EXTS):
            vids.append(fullpath)

    return sorted(vids)


def make_output_path(video_path):
    base = os.path.basename(video_path)
    stem, ext = os.path.splitext(base)
    return os.path.join(OUTPUT_FOLDER, f"{stem}_tracked{ext}")


def make_speed_report_path(video_path):
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    return os.path.join(OUTPUT_FOLDER, f"{stem}_speeds.txt")


def estimate_speed_mph(track_history, fps, feet_per_pixel):
    if len(track_history) < MIN_SPEED_SAMPLE_FRAMES or fps <= 0:
        return None

    start_frame, start_x = track_history[0]
    end_frame, end_x = track_history[-1]
    delta_frames = end_frame - start_frame

    if delta_frames <= 0:
        return None

    delta_pixels = abs(end_x - start_x)
    if delta_pixels < MIN_HORIZONTAL_TRAVEL_PIXELS:
        return None

    delta_feet = delta_pixels * feet_per_pixel
    delta_seconds = delta_frames / fps
    feet_per_second = delta_feet / delta_seconds
    return feet_per_second * 3600.0 / 5280.0


def get_track_color(current_speed_mph, previous_speed_mph):
    if current_speed_mph is None or previous_speed_mph is None:
        return (0, 255, 255)

    speed_delta = current_speed_mph - previous_speed_mph
    if speed_delta > SPEED_CHANGE_THRESHOLD_MPH:
        return (0, 255, 0)
    if speed_delta < -SPEED_CHANGE_THRESHOLD_MPH:
        return (0, 0, 255)
    return (0, 255, 255)


def draw_overlay(frame, frame_num, frame_time, raw_dets, active_tracks, total_unique, avg_speed_mph):
    cv2.putText(frame, f"Frame: {frame_num}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Time: {frame_time:.4f}s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Raw Detections: {raw_dets}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Active Tracks: {active_tracks}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Unique Vehicles: {total_unique}", (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    avg_speed_text = f"{avg_speed_mph:.1f} mph" if avg_speed_mph is not None else "--"
    cv2.putText(frame, f"Avg Active Speed: {avg_speed_text}", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def write_speed_report(report_path, video_path, fps, frame_count, track_speed_samples):
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(f"Video: {os.path.basename(video_path)}\n")
        report_file.write(f"FPS: {fps:.4f}\n")
        report_file.write(f"Frames Processed: {frame_count}\n")
        report_file.write(f"Horizontal Calibration: {SCREEN_WIDTH_FEET:.1f} ft across frame width\n")
        report_file.write("\n")

        if not track_speed_samples:
            report_file.write("No speed samples were recorded.\n")
            return

        report_file.write("Per-Car Summary\n")
        report_file.write("================\n")
        for track_id in sorted(track_speed_samples):
            samples = track_speed_samples[track_id]
            speeds = [sample["speed_mph"] for sample in samples]
            avg_speed = sum(speeds) / len(speeds)
            last_sample = samples[-1]
            report_file.write(
                f"ID {track_id}: "
                f"samples={len(samples)}, "
                f"avg={avg_speed:.2f} mph, "
                f"max={max(speeds):.2f} mph, "
                f"last={last_sample['speed_mph']:.2f} mph "
                f"(frame {last_sample['frame']}, time {last_sample['video_time_s']:.2f}s)\n"
            )

        report_file.write("\nDetailed Samples\n")
        report_file.write("================\n")
        for track_id in sorted(track_speed_samples):
            report_file.write(f"ID {track_id}\n")
            for sample in track_speed_samples[track_id]:
                report_file.write(
                    f"  frame={sample['frame']}, "
                    f"time={sample['video_time_s']:.2f}s, "
                    f"speed={sample['speed_mph']:.2f} mph\n"
                )
            report_file.write("\n")


def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    out_path = make_output_path(video_path)
    speed_report_path = make_speed_report_path(video_path)

    out_writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    if not out_writer.isOpened():
        print(f"Could not create output video: {out_path}")
        cap.release()
        return

    seen_ids = set()
    track_histories = defaultdict(lambda: deque(maxlen=SPEED_WINDOW_FRAMES))
    track_speeds_mph = {}
    previous_track_speeds_mph = {}
    track_speed_samples = defaultdict(list)
    frame_num = 0
    total_time = 0.0
    stop_requested = False
    # Horizontal calibration: the full visible width is treated as 500 feet.
    feet_per_pixel = SCREEN_WIDTH_FEET / max(w, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            t0 = time.perf_counter()

            results = model.track(
                frame,
                tracker="bytetrack.yaml",
                conf=CONF_THRESH,
                imgsz=IMG_SIZE,
                classes=VEHICLE_CLASSES,
                persist=True,
                verbose=False
            )

            result = results[0]
            plotted = frame.copy()

            raw_dets = 0
            active_tracks = 0
            active_speed_samples = []

            if result.boxes is not None and len(result.boxes) > 0:
                raw_dets = len(result.boxes)

                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                ids = None
                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()

                for i, box in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = box.astype(int)
                    conf = confs[i]

                    track_id = None
                    if ids is not None and i < len(ids):
                        track_id = ids[i]

                    # count active tracks only when id exists
                    if track_id is not None:
                        active_tracks += 1
                        track_id = int(track_id)
                        seen_ids.add(track_id)

                        center_x = (x1 + x2) / 2.0
                        track_histories[track_id].append((frame_num, center_x))

                        speed_mph = estimate_speed_mph(
                            track_histories[track_id],
                            fps,
                            feet_per_pixel
                        )
                        previous_speed_mph = track_speeds_mph.get(track_id)
                        if speed_mph is not None:
                            track_speeds_mph[track_id] = speed_mph
                            previous_track_speeds_mph[track_id] = previous_speed_mph
                            track_speed_samples[track_id].append({
                                "frame": frame_num,
                                "video_time_s": frame_num / fps,
                                "speed_mph": speed_mph,
                            })

                        speed_mph = track_speeds_mph.get(track_id)
                        previous_speed_mph = previous_track_speeds_mph.get(track_id)
                        if speed_mph is not None:
                            active_speed_samples.append(speed_mph)
                            label = f"ID {track_id} | {speed_mph:.1f} mph"
                        else:
                            label = f"ID {track_id} | car {conf:.2f}"
                        color = get_track_color(speed_mph, previous_speed_mph)
                    else:
                        label = f"car {conf:.2f}"
                        color = (0, 165, 255)

                    cv2.rectangle(plotted, (x1, y1), (x2, y2), color, LINE_WIDTH)

                    y_text = y1 - 8 if y1 - 8 > 10 else y1 + 20
                    cv2.putText(plotted, label, (x1, y_text),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            t1 = time.perf_counter()
            frame_time = t1 - t0
            total_time += frame_time
            avg_speed_mph = (
                sum(active_speed_samples) / len(active_speed_samples)
                if active_speed_samples else None
            )

            draw_overlay(
                plotted,
                frame_num,
                frame_time,
                raw_dets,
                active_tracks,
                len(seen_ids),
                avg_speed_mph
            )

            print(f"Frame {frame_num} | Vehicles tracked: {active_tracks}", flush=True)

            avg_speed_text = f"{avg_speed_mph:.2f} mph" if avg_speed_mph is not None else "--"
            print(
                f"{os.path.basename(video_path)} | "
                f"frame {frame_num} | "
                f"time {frame_time:.6f} s | "
                f"raw {raw_dets} | "
                f"active {active_tracks} | "
                f"total {len(seen_ids)} | "
                f"avg speed {avg_speed_text}",
                flush=True
            )

            out_writer.write(plotted)

            if frame_num == 1 or frame_num % REPORT_FLUSH_EVERY_FRAMES == 0:
                write_speed_report(speed_report_path, video_path, fps, frame_num, track_speed_samples)

            if SHOW_VIDEO:
                cv2.imshow("Tracking", plotted)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    stop_requested = True
                    break
    except KeyboardInterrupt:
        stop_requested = True
        print("\nStop requested. Finalizing current video...", flush=True)
    finally:
        cap.release()
        out_writer.release()
        write_speed_report(speed_report_path, video_path, fps, frame_num, track_speed_samples)

    avg_time = total_time / frame_num if frame_num > 0 else 0.0

    status_text = "Stopped early" if stop_requested else "Finished"
    print(f"\n{status_text}: {os.path.basename(video_path)}")
    print(f"Frames: {frame_num}")
    print(f"Avg frame time: {avg_time:.6f} s")
    print(f"Total vehicles: {len(seen_ids)}")
    print(f"Saved to: {out_path}\n")
    print(f"Speed report: {speed_report_path}\n")
    return not stop_requested


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Missing input folder: {INPUT_FOLDER}")
        return

    if not os.path.isfile(MODEL_WEIGHTS):
        print(f"Missing model weights: {MODEL_WEIGHTS}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    videos = collect_videos(INPUT_FOLDER)

    if not videos:
        print("No videos found in input_videos/")
        return

    model = YOLO(MODEL_WEIGHTS)

    for video_path in videos:
        completed = process_video(model, video_path)
        if not completed:
            break

    if SHOW_VIDEO:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
