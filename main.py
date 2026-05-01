import os
import cv2
import time
import csv
import tempfile
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "out_videos"
LANE_ANNOTATION_XML = "configs/bound_box_lanes.xml"

MODEL_WEIGHTS = "runs/detect/runs_highway/car_detector_ft/weights/best.pt"
# custom model has one class only: car
VEHICLE_CLASSES = [0]

CONF_THRESH = 0.03
IMG_SIZE = 1920
LINE_WIDTH = 2
LANE_LINE_WIDTH = 1

SHOW_VIDEO = False

SCREEN_WIDTH_FEET = 500.0
SPEED_ESTIMATE_FRAME_GAP = 4
SPEED_ESTIMATE_HISTORY_POINTS = 6
RAW_SPEED_HISTORY_SIZE = 5
MIN_CENTROID_TRAVEL_PIXELS = 8.0
SPEED_SMOOTHING_ALPHA = 0.2
EDGE_MARGIN_PIXELS = 3
ACCELERATION_THRESHOLD_MPH_PER_10_FRAMES = 2.0
MAX_COLOR_RATE_MPH_PER_10_FRAMES = 6.0
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


def make_metrics_csv_path(video_path):
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    return os.path.join(OUTPUT_FOLDER, f"{stem}_car_metrics.csv")


def parse_lane_number(label):
    parts = label.strip().split()
    if len(parts) != 2 or parts[0].lower() != "lane":
        raise ValueError(f"Unsupported lane label: {label}")
    return int(parts[1])


def load_lane_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image = root.find("image")
    if image is None:
        raise RuntimeError(f"No <image> section found in {xml_path}")

    lane_boxes = []
    for box in image.findall("box"):
        label = box.attrib["label"]
        lane_boxes.append({
            "lane_number": parse_lane_number(label),
            "label": label,
            "xtl": float(box.attrib["xtl"]),
            "ytl": float(box.attrib["ytl"]),
            "xbr": float(box.attrib["xbr"]),
            "ybr": float(box.attrib["ybr"]),
        })

    if not lane_boxes:
        raise RuntimeError(f"No lane boxes found in {xml_path}")

    lane_boxes.sort(key=lambda lane: lane["lane_number"])
    for index, lane in enumerate(lane_boxes, start=1):
        if lane["lane_number"] != index:
            raise RuntimeError("Lane numbers must be consecutive starting at 1.")

    return lane_boxes


def intersect_area(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    if right <= left or bottom <= top:
        return 0.0

    return (right - left) * (bottom - top)


def get_majority_lane(box_xyxy, lane_boxes):
    x1, y1, x2, y2 = box_xyxy
    box_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if box_area <= 0:
        return None

    best_lane = None
    best_area = 0.0

    for lane in lane_boxes:
        lane_rect = (lane["xtl"], lane["ytl"], lane["xbr"], lane["ybr"])
        overlap_area = intersect_area((x1, y1, x2, y2), lane_rect)
        if overlap_area > best_area:
            best_area = overlap_area
            best_lane = lane["lane_number"]

    if best_area > box_area / 2.0:
        return best_lane

    return None


def draw_lane_overlay(frame, lane_boxes):
    for lane in lane_boxes:
        x1 = int(round(lane["xtl"]))
        y1 = int(round(lane["ytl"]))
        x2 = int(round(lane["xbr"]))
        y2 = int(round(lane["ybr"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), LANE_LINE_WIDTH)
        cv2.putText(
            frame,
            lane["label"],
            (x1 + 10, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )


def initialize_track_stats():
    return {
        "first_frame": None,
        "last_frame": None,
        "first_time_s": None,
        "last_time_s": None,
        "active_frame_count": 0,
        "speed_samples_mph": [],
        "lane_frames": defaultdict(int),
    }


def interpolate_bgr(color_a, color_b, ratio):
    ratio = min(max(ratio, 0.0), 1.0)
    return tuple(
        int(round(channel_a + (channel_b - channel_a) * ratio))
        for channel_a, channel_b in zip(color_a, color_b)
    )


def build_metrics_rows(fps, lane_boxes, track_stats):
    lane_numbers = [lane["lane_number"] for lane in lane_boxes]
    fieldnames = [
        "tracker_id",
        "initial_speed_mph",
        "final_speed_mph",
        "avg_speed_mph",
        "start_time_s",
        "time_in_frame_s",
        "final_time_s",
    ] + [f"lane_{lane_number}_time_s" for lane_number in lane_numbers]

    rows = []
    for track_id in sorted(track_stats):
        stats = track_stats[track_id]
        first_frame = stats["first_frame"]
        last_frame = stats["last_frame"]

        if first_frame is None or last_frame is None:
            continue

        speeds = stats["speed_samples_mph"]
        row = {
            "tracker_id": track_id,
            "initial_speed_mph": f"{speeds[0]:.4f}" if speeds else "",
            "final_speed_mph": f"{speeds[-1]:.4f}" if speeds else "",
            "avg_speed_mph": f"{(sum(speeds) / len(speeds)):.4f}" if speeds else "",
            "start_time_s": f"{stats['first_time_s']:.4f}",
            "time_in_frame_s": f"{(stats['active_frame_count'] / fps):.4f}",
            "final_time_s": f"{stats['last_time_s']:.4f}",
        }

        for lane_number in lane_numbers:
            lane_frame_count = stats["lane_frames"].get(lane_number, 0)
            row[f"lane_{lane_number}_time_s"] = f"{(lane_frame_count / fps):.4f}"

        rows.append(row)

    return fieldnames, rows


def make_locked_fallback_path(csv_path):
    folder = os.path.dirname(csv_path)
    stem, ext = os.path.splitext(os.path.basename(csv_path))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(folder, f"{stem}_live_{timestamp}{ext}")
    suffix = 1

    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem}_live_{timestamp}_{suffix}{ext}")
        suffix += 1

    return candidate


def write_csv_atomic(csv_path, fieldnames, rows):
    folder = os.path.dirname(csv_path) or "."
    os.makedirs(folder, exist_ok=True)
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            delete=False,
            dir=folder,
            prefix="tmp_metrics_",
            suffix=".csv",
        ) as temp_file:
            temp_path = temp_file.name
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temp_path, csv_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def write_car_metrics_csv(csv_path, fps, lane_boxes, track_stats):
    fieldnames, rows = build_metrics_rows(fps, lane_boxes, track_stats)

    try:
        write_csv_atomic(csv_path, fieldnames, rows)
        return csv_path, False
    except PermissionError:
        fallback_path = make_locked_fallback_path(csv_path)
        write_csv_atomic(fallback_path, fieldnames, rows)
        return fallback_path, True


def is_box_near_frame_edge(box_xyxy, frame_width, frame_height, margin_pixels):
    x1, y1, x2, y2 = box_xyxy
    return (
        x1 <= margin_pixels
        or y1 <= margin_pixels
        or x2 >= frame_width - margin_pixels
        or y2 >= frame_height - margin_pixels
    )


def estimate_speed_mph(track_history, fps, feet_per_pixel):
    if len(track_history) < 2 or fps <= 0:
        return None

    start_frame, start_center = track_history[0]
    end_frame, end_center = track_history[-1]
    delta_frames = end_frame - start_frame

    if delta_frames <= 0:
        return None

    # Cars in this view move predominantly left/right, so horizontal centroid
    # motion is more stable than full 2D box jitter.
    delta_pixels = abs(end_center[0] - start_center[0])
    if delta_pixels < MIN_CENTROID_TRAVEL_PIXELS:
        return None

    delta_feet = delta_pixels * feet_per_pixel
    delta_seconds = delta_frames / fps
    feet_per_second = delta_feet / delta_seconds
    return feet_per_second * 3600.0 / 5280.0


def smooth_speed_mph(raw_speed_history, previous_speed_mph):
    if not raw_speed_history:
        return None

    median_speed_mph = statistics.median(raw_speed_history)
    if previous_speed_mph is None:
        return median_speed_mph

    return (
        SPEED_SMOOTHING_ALPHA * median_speed_mph
        + (1.0 - SPEED_SMOOTHING_ALPHA) * previous_speed_mph
    )


def get_track_color(current_speed_mph, previous_speed_mph, current_frame, previous_frame):
    if current_speed_mph is None or previous_speed_mph is None:
        return (0, 255, 255)

    if current_frame is None or previous_frame is None:
        return (0, 255, 255)

    delta_frames = current_frame - previous_frame
    if delta_frames <= 0:
        return (0, 255, 255)

    speed_delta = current_speed_mph - previous_speed_mph
    rate_per_10_frames = speed_delta * 10.0 / delta_frames
    abs_rate = abs(rate_per_10_frames)

    if abs_rate <= ACCELERATION_THRESHOLD_MPH_PER_10_FRAMES:
        return (0, 255, 255)

    blend_span = max(
        MAX_COLOR_RATE_MPH_PER_10_FRAMES - ACCELERATION_THRESHOLD_MPH_PER_10_FRAMES,
        1e-6,
    )
    ratio = min(
        (abs_rate - ACCELERATION_THRESHOLD_MPH_PER_10_FRAMES) / blend_span,
        1.0,
    )

    if rate_per_10_frames > 0:
        return interpolate_bgr((0, 255, 255), (0, 255, 0), ratio)

    return interpolate_bgr((0, 255, 255), (0, 0, 255), ratio)


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
    metrics_csv_path = make_metrics_csv_path(video_path)
    lane_boxes = load_lane_boxes(LANE_ANNOTATION_XML)
    metrics_path_was_redirected = False

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
    track_histories = defaultdict(lambda: deque(maxlen=SPEED_ESTIMATE_HISTORY_POINTS))
    track_speeds_mph = {}
    previous_track_speeds_mph = {}
    previous_speed_sample_frames = {}
    current_speed_sample_frames = {}
    raw_speed_histories = defaultdict(lambda: deque(maxlen=RAW_SPEED_HISTORY_SIZE))
    track_speed_samples = defaultdict(list)
    track_stats = defaultdict(initialize_track_stats)
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
            draw_lane_overlay(plotted, lane_boxes)

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
                        center_y = (y1 + y2) / 2.0

                        current_time_s = frame_num / fps
                        stats = track_stats[track_id]
                        if stats["first_frame"] is None:
                            stats["first_frame"] = frame_num
                            stats["first_time_s"] = (frame_num - 1) / fps
                        stats["last_frame"] = frame_num
                        stats["last_time_s"] = current_time_s
                        stats["active_frame_count"] += 1

                        majority_lane = get_majority_lane(
                            (float(x1), float(y1), float(x2), float(y2)),
                            lane_boxes,
                        )
                        if majority_lane is not None:
                            stats["lane_frames"][majority_lane] += 1

                        speed_mph = None
                        previous_speed_mph = track_speeds_mph.get(track_id)
                        box_near_edge = is_box_near_frame_edge(
                            (x1, y1, x2, y2),
                            w,
                            h,
                            EDGE_MARGIN_PIXELS,
                        )

                        if not box_near_edge:
                            history = track_histories[track_id]
                            if not history or frame_num - history[-1][0] >= SPEED_ESTIMATE_FRAME_GAP:
                                history.append((frame_num, (center_x, center_y)))

                            raw_speed_mph = estimate_speed_mph(
                                history,
                                fps,
                                feet_per_pixel
                            )

                            if raw_speed_mph is not None:
                                raw_speed_histories[track_id].append(raw_speed_mph)
                                speed_mph = smooth_speed_mph(
                                    raw_speed_histories[track_id],
                                    previous_speed_mph,
                                )

                        if speed_mph is not None:
                            track_speeds_mph[track_id] = speed_mph
                            previous_track_speeds_mph[track_id] = previous_speed_mph if previous_speed_mph is not None else speed_mph
                            previous_speed_sample_frames[track_id] = track_speed_samples[track_id][-1]["frame"] if track_speed_samples[track_id] else frame_num
                            current_speed_sample_frames[track_id] = frame_num
                            track_speed_samples[track_id].append({
                                "frame": frame_num,
                                "video_time_s": current_time_s,
                                "speed_mph": speed_mph,
                            })
                            stats["speed_samples_mph"].append(speed_mph)

                        speed_mph = track_speeds_mph.get(track_id)
                        previous_speed_mph = previous_track_speeds_mph.get(track_id)
                        if speed_mph is not None:
                            active_speed_samples.append(speed_mph)
                            label = f"ID {track_id} | {speed_mph:.1f} mph"
                        else:
                            label = f"ID {track_id} | car {conf:.2f}"

                        if majority_lane is not None:
                            label += f" | Lane {majority_lane}"
                        color = get_track_color(
                            speed_mph,
                            previous_speed_mph,
                            current_speed_sample_frames.get(track_id),
                            previous_speed_sample_frames.get(track_id),
                        )
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
                metrics_csv_path, metrics_path_was_redirected = write_car_metrics_csv(
                    metrics_csv_path,
                    fps,
                    lane_boxes,
                    track_stats,
                )
                if metrics_path_was_redirected and frame_num == 1:
                    print(
                        f"Metrics CSV target was locked. Writing to fallback path: {metrics_csv_path}",
                        flush=True,
                    )

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
        metrics_csv_path, metrics_path_was_redirected = write_car_metrics_csv(
            metrics_csv_path,
            fps,
            lane_boxes,
            track_stats,
        )

    avg_time = total_time / frame_num if frame_num > 0 else 0.0

    status_text = "Stopped early" if stop_requested else "Finished"
    print(f"\n{status_text}: {os.path.basename(video_path)}")
    print(f"Frames: {frame_num}")
    print(f"Avg frame time: {avg_time:.6f} s")
    print(f"Total vehicles: {len(seen_ids)}")
    print(f"Saved to: {out_path}\n")
    print(f"Speed report: {speed_report_path}")
    print(f"Car metrics CSV: {metrics_csv_path}\n")
    return not stop_requested


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Missing input folder: {INPUT_FOLDER}")
        return

    if not os.path.isfile(MODEL_WEIGHTS):
        print(f"Missing model weights: {MODEL_WEIGHTS}")
        return

    if not os.path.isfile(LANE_ANNOTATION_XML):
        print(f"Missing lane annotation file: {LANE_ANNOTATION_XML}")
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
