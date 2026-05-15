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

MODEL_WEIGHTS = "runs_highway/visdrone_vehicles_ft/weights/best.pt"
# Size-focused vehicle model classes trained from VisDrone.
VEHICLE_CLASS_NAMES = {
    0: "car",
    1: "big_car",
    2: "motorcycle",
}
VEHICLE_CLASSES = sorted(VEHICLE_CLASS_NAMES)

CONF_THRESH = 0.03
IMG_SIZE = 1920
LINE_WIDTH = 2
LANE_LINE_WIDTH = 1
VEHICLE_TAG_TEXT_COLOR = (255, 255, 255)
VEHICLE_TAG_MUTED_TEXT_COLOR = (220, 245, 255)
VEHICLE_TAG_BG_COLOR = (10, 18, 26)
VEHICLE_TAG_BORDER_COLOR = (255, 255, 255)
VEHICLE_TAG_LEADER_LINE_COLOR = (230, 230, 230)
VEHICLE_TAG_BG_ALPHA = 0.82
VEHICLE_TAG_FONT_SCALE = 0.42
VEHICLE_TAG_FONT_THICKNESS = 1
VEHICLE_TAG_PADDING_X = 5
VEHICLE_TAG_PADDING_Y = 3
VEHICLE_TAG_LINE_GAP = 2
VEHICLE_TAG_GAP = 4
VEHICLE_TAG_MIN_GAP = 3

SHOW_VIDEO = True

SCREEN_WIDTH_FEET = 417.4
VEHICLE_TRAVEL_DIRECTION = "left"
NO_NUMERIC_VALUE = "NaN"
AMPLE_MERGE_GAP_CAR_LENGTHS = 2.0
MERGE_LANE_CONFIRMATION_FRAMES = 3
LANE_TOUCH_MIN_OVERLAP_RATIO = 0.02
SPEED_ESTIMATE_FRAME_GAP = 1
SPEED_ESTIMATE_HISTORY_POINTS = 16
RAW_SPEED_HISTORY_SIZE = 9
MIN_SPEED_HISTORY_POINTS = 6
MIN_CENTROID_TRAVEL_PIXELS = 10.0
SPEED_SMOOTHING_ALPHA = 0.15
EDGE_MARGIN_PIXELS = 80
NON_EDGE_WARMUP_FRAMES = 8
ACCELERATION_HISTORY_SIZE = 14
MIN_ACCELERATION_HISTORY_POINTS = 6
ACCELERATION_SMOOTHING_ALPHA = 0.20
ACCELERATION_THRESHOLD_MPH_PER_10_FRAMES = 3.0
MAX_COLOR_RATE_MPH_PER_10_FRAMES = 10.0
REPORT_FLUSH_EVERY_FRAMES = 30

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")
OUTPUT_ARTIFACT_SPECS = {
    "video": ("tracked", None),
    "speed_report": ("speeds", ".txt"),
    "metrics_csv": ("vehicle_metrics", ".csv"),
    "telemetry_csv": ("vehicle_telemetry", ".csv"),
    "merge_events_csv": ("merge_events", ".csv"),
    "following_graph": ("following_distance", ".png"),
}


# =========================================================
# HELPERS
# =========================================================

def collect_videos(folder):
    if not os.path.isdir(folder):
        return []

    return sorted(
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, filename))
        and filename.endswith(VIDEO_EXTS)
    )


def make_output_artifact_path(video_path, suffix, extension=None):
    stem, source_ext = os.path.splitext(os.path.basename(video_path))
    return os.path.join(OUTPUT_FOLDER, f"{stem}_{suffix}{extension or source_ext}")


def make_output_paths(video_path):
    return {
        key: make_output_artifact_path(video_path, suffix, extension)
        for key, (suffix, extension) in OUTPUT_ARTIFACT_SPECS.items()
    }


def get_vehicle_class_name(class_id):
    return VEHICLE_CLASS_NAMES.get(int(class_id), "vehicle")


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
            "source_label": label,
            "label": label,
            "xtl": float(box.attrib["xtl"]),
            "ytl": float(box.attrib["ytl"]),
            "xbr": float(box.attrib["xbr"]),
            "ybr": float(box.attrib["ybr"]),
        })

    if not lane_boxes:
        raise RuntimeError(f"No lane boxes found in {xml_path}")

    # In this camera view, the leftmost lane in road coordinates is the bottom
    # lane in image coordinates. Renumber bottom-to-top so Lane 1 is leftmost.
    lane_boxes.sort(
        key=lambda lane: (lane["ytl"] + lane["ybr"]) / 2.0,
        reverse=True,
    )
    for index, lane in enumerate(lane_boxes, start=1):
        lane["source_lane_number"] = lane["lane_number"]
        lane["lane_number"] = index
        lane["label"] = f"Lane {index}"

    return lane_boxes


def intersect_area(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    if right <= left or bottom <= top:
        return 0.0

    return (right - left) * (bottom - top)


def get_box_area(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def get_lane_contacts(box_xyxy, lane_boxes):
    box_area = get_box_area(box_xyxy)
    if box_area <= 0:
        return []

    contacts = []
    for lane in lane_boxes:
        lane_rect = (lane["xtl"], lane["ytl"], lane["xbr"], lane["ybr"])
        overlap_area = intersect_area(box_xyxy, lane_rect)
        overlap_ratio = overlap_area / box_area
        if overlap_ratio >= LANE_TOUCH_MIN_OVERLAP_RATIO:
            contacts.append({
                "lane": lane["lane_number"],
                "overlap_area_px": overlap_area,
                "overlap_ratio": overlap_ratio,
            })

    contacts.sort(key=lambda contact: contact["overlap_ratio"], reverse=True)
    return contacts


def get_lane_box_by_number(lane_boxes, lane_number):
    for lane in lane_boxes:
        if lane["lane_number"] == lane_number:
            return lane
    return None


def get_lane_boundary_y_px(lane_boxes, lane_a, lane_b):
    lane_box_a = get_lane_box_by_number(lane_boxes, lane_a)
    lane_box_b = get_lane_box_by_number(lane_boxes, lane_b)
    if lane_box_a is None or lane_box_b is None:
        return None

    center_a = (lane_box_a["ytl"] + lane_box_a["ybr"]) / 2.0
    center_b = (lane_box_b["ytl"] + lane_box_b["ybr"]) / 2.0
    if center_a > center_b:
        lower_lane = lane_box_a
        upper_lane = lane_box_b
    else:
        lower_lane = lane_box_b
        upper_lane = lane_box_a

    return (lower_lane["ytl"] + upper_lane["ybr"]) / 2.0


def get_lane_contact_ratio(lane_contacts, lane_number):
    for contact in lane_contacts:
        if contact["lane"] == lane_number:
            return contact["overlap_ratio"]
    return None


def format_lane_contact_lanes(lane_contacts):
    return "|".join(str(contact["lane"]) for contact in lane_contacts)


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
        "vehicle_type": None,
        "first_frame": None,
        "last_frame": None,
        "first_time_s": None,
        "last_time_s": None,
        "active_frame_count": 0,
        "speed_samples_mph": [],
        "acceleration_samples_mph_per_10_frames": [],
        "following_distance_samples_ft": [],
        "following_time_samples_s": [],
        "merge_event_count": 0,
        "ample_merge_event_count": 0,
        "tight_merge_event_count": 0,
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
        "vehicle_type",
        "initial_speed_mph",
        "final_speed_mph",
        "avg_speed_mph",
        "avg_acceleration_mph_per_10_frames",
        "min_following_distance_ft",
        "avg_following_distance_ft",
        "min_following_time_s",
        "avg_following_time_s",
        "merge_event_count",
        "ample_merge_event_count",
        "tight_merge_event_count",
        "start_time_s",
        "time_in_frame_s",
        "final_time_s",
    ] + [f"lane_{lane_number}_time_s" for lane_number in lane_numbers]

    rows = []
    for track_id in sorted(track_stats):
        stats = track_stats[track_id]
        first_frame = stats["first_frame"]
        last_frame = stats["last_frame"]
        speeds = stats["speed_samples_mph"]
        acceleration_samples = stats["acceleration_samples_mph_per_10_frames"]
        following_distances = stats["following_distance_samples_ft"]
        following_times = stats["following_time_samples_s"]

        if first_frame is None or last_frame is None:
            continue

        row = {
            "tracker_id": track_id,
            "vehicle_type": stats["vehicle_type"] or "",
            "initial_speed_mph": f"{speeds[0]:.4f}" if speeds else "",
            "final_speed_mph": f"{speeds[-1]:.4f}" if speeds else "",
            "avg_speed_mph": f"{(sum(speeds) / len(speeds)):.4f}" if speeds else "",
            "avg_acceleration_mph_per_10_frames": (
                f"{(sum(acceleration_samples) / len(acceleration_samples)):.4f}"
                if acceleration_samples else ""
            ),
            "min_following_distance_ft": (
                f"{min(following_distances):.4f}" if following_distances else ""
            ),
            "avg_following_distance_ft": (
                f"{(sum(following_distances) / len(following_distances)):.4f}"
                if following_distances else ""
            ),
            "min_following_time_s": (
                f"{min(following_times):.4f}" if following_times else ""
            ),
            "avg_following_time_s": (
                f"{(sum(following_times) / len(following_times)):.4f}"
                if following_times else ""
            ),
            "merge_event_count": stats["merge_event_count"],
            "ample_merge_event_count": stats["ample_merge_event_count"],
            "tight_merge_event_count": stats["tight_merge_event_count"],
            "start_time_s": f"{stats['first_time_s']:.4f}",
            "time_in_frame_s": f"{(stats['active_frame_count'] / fps):.4f}",
            "final_time_s": f"{stats['last_time_s']:.4f}",
        }

        for lane_number in lane_numbers:
            lane_frame_count = stats["lane_frames"].get(lane_number, 0)
            row[f"lane_{lane_number}_time_s"] = f"{(lane_frame_count / fps):.4f}"

        rows.append(row)

    return fieldnames, rows


TELEMETRY_FIELDNAMES = [
    "frame",
    "video_time_s",
    "tracker_id",
    "vehicle_type",
    "lane",
    "center_x_px",
    "center_y_px",
    "center_x_ft",
    "center_y_ft",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "speed_mph",
    "acceleration_mph_per_10_frames",
    "motion_direction",
    "leader_tracker_id",
    "following_status",
    "following_gap_px",
    "following_gap_car_lengths",
    "following_distance_ft",
    "following_time_s",
    "front_clearance_to_frame_edge_px",
    "stable_lane",
    "merge_event",
    "merge_from_lane",
    "merge_to_lane",
    "merge_front_vehicle_id",
    "merge_front_gap_px",
    "merge_front_gap_car_lengths",
    "merge_rear_vehicle_id",
    "merge_rear_gap_px",
    "merge_rear_gap_car_lengths",
    "merge_space_status",
    "merge_has_ample_space",
    "lane_boundary_contact",
    "lane_boundary_event_id",
    "lane_boundary_from_lane",
    "lane_boundary_to_lane",
    "lane_boundary_y_px",
    "lane_contact_lanes",
]


MERGE_EVENT_FIELDNAMES = [
    "event_id",
    "tracker_id",
    "vehicle_type",
    "from_lane",
    "to_lane",
    "boundary_y_px",
    "start_frame",
    "start_time_s",
    "start_center_x_px",
    "start_center_y_px",
    "start_bbox_x1",
    "start_bbox_y1",
    "start_bbox_x2",
    "start_bbox_y2",
    "start_from_lane_overlap_ratio",
    "start_to_lane_overlap_ratio",
    "start_front_vehicle_id",
    "start_front_gap_px",
    "start_front_gap_car_lengths",
    "start_rear_vehicle_id",
    "start_rear_gap_px",
    "start_rear_gap_car_lengths",
    "end_frame",
    "end_time_s",
    "end_center_x_px",
    "end_center_y_px",
    "end_bbox_x1",
    "end_bbox_y1",
    "end_bbox_x2",
    "end_bbox_y2",
    "end_lane",
    "end_stable_lane",
    "end_from_lane_overlap_ratio",
    "end_to_lane_overlap_ratio",
    "end_front_vehicle_id",
    "end_front_gap_px",
    "end_front_gap_car_lengths",
    "end_rear_vehicle_id",
    "end_rear_gap_px",
    "end_rear_gap_car_lengths",
    "duration_frames",
    "duration_s",
    "end_reason",
]


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


def write_csv_with_fallback(csv_path, fieldnames, rows):
    try:
        write_csv_atomic(csv_path, fieldnames, rows)
        return csv_path, False
    except PermissionError:
        fallback_path = make_locked_fallback_path(csv_path)
        write_csv_atomic(fallback_path, fieldnames, rows)
        return fallback_path, True


def write_vehicle_metrics_csv(csv_path, fps, lane_boxes, track_stats):
    fieldnames, rows = build_metrics_rows(fps, lane_boxes, track_stats)
    return write_csv_with_fallback(csv_path, fieldnames, rows)


def write_vehicle_telemetry_csv(csv_path, telemetry_rows):
    return write_csv_with_fallback(csv_path, TELEMETRY_FIELDNAMES, telemetry_rows)


def write_merge_events_csv(csv_path, merge_event_rows):
    return write_csv_with_fallback(csv_path, MERGE_EVENT_FIELDNAMES, merge_event_rows)


def write_following_distance_graph(graph_path, telemetry_rows):
    graph_rows = [
        row for row in telemetry_rows
        if row.get("following_time_s") not in ("", None, NO_NUMERIC_VALUE)
    ]

    if not graph_rows:
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping following-distance graph.", flush=True)
        return False

    rows_by_track = defaultdict(list)
    for row in graph_rows:
        rows_by_track[row["tracker_id"]].append(row)

    os.makedirs(os.path.dirname(graph_path) or ".", exist_ok=True)
    plt.figure(figsize=(14, 8))

    for track_id in sorted(rows_by_track):
        rows = sorted(rows_by_track[track_id], key=lambda item: float(item["video_time_s"]))
        times = [float(row["video_time_s"]) for row in rows]
        following_times = [float(row["following_time_s"]) for row in rows]
        vehicle_type = rows[0]["vehicle_type"]
        plt.plot(
            times,
            following_times,
            linewidth=1.2,
            alpha=0.75,
            label=f"ID {track_id} {vehicle_type}",
        )

    plt.axhline(2.0, color="red", linestyle="--", linewidth=1.0, label="2 second reference")
    plt.title("Following Distance Over Time")
    plt.xlabel("Video time (s)")
    plt.ylabel("Following distance (s)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(graph_path, dpi=160)
    plt.close()
    return True


def is_box_near_frame_edge(box_xyxy, frame_width, frame_height, margin_pixels):
    x1, y1, x2, y2 = box_xyxy
    return (
        x1 <= margin_pixels
        or y1 <= margin_pixels
        or x2 >= frame_width - margin_pixels
        or y2 >= frame_height - margin_pixels
    )


def linear_regression_slope(x_values, y_values):
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None

    avg_x = sum(x_values) / len(x_values)
    avg_y = sum(y_values) / len(y_values)
    denominator = sum((x - avg_x) ** 2 for x in x_values)

    if denominator <= 0:
        return None

    numerator = sum(
        (x - avg_x) * (y - avg_y)
        for x, y in zip(x_values, y_values)
    )
    return numerator / denominator


def estimate_motion_slope_pixels_per_frame(track_history, min_points=2):
    if len(track_history) < min_points:
        return None

    frame_values = [float(frame) for frame, _ in track_history]
    center_x_values = [float(center[0]) for _, center in track_history]
    return linear_regression_slope(frame_values, center_x_values)


def estimate_speed_mph(track_history, fps, feet_per_pixel):
    if len(track_history) < MIN_SPEED_HISTORY_POINTS or fps <= 0:
        return None

    start_frame = track_history[0][0]
    end_frame = track_history[-1][0]
    delta_frames = end_frame - start_frame

    if delta_frames <= 0:
        return None

    # Cars in this view move predominantly left/right, so horizontal centroid
    # motion is more stable than full 2D box jitter. Fit all recent samples so
    # one bad box does not dominate the estimate.
    slope_pixels_per_frame = estimate_motion_slope_pixels_per_frame(
        track_history,
        min_points=MIN_SPEED_HISTORY_POINTS,
    )

    if slope_pixels_per_frame is None:
        return None

    delta_pixels = abs(slope_pixels_per_frame) * delta_frames
    if delta_pixels < MIN_CENTROID_TRAVEL_PIXELS:
        return None

    feet_per_second = abs(slope_pixels_per_frame) * feet_per_pixel * fps
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


def estimate_speed_rate_mph_per_10_frames(speed_history):
    if len(speed_history) < MIN_ACCELERATION_HISTORY_POINTS:
        return None

    frame_values = [float(frame) for frame, _ in speed_history]
    speed_values = [float(speed_mph) for _, speed_mph in speed_history]
    slope_mph_per_frame = linear_regression_slope(frame_values, speed_values)

    if slope_mph_per_frame is None:
        return None

    return slope_mph_per_frame * 10.0


def smooth_speed_rate_mph_per_10_frames(raw_rate, previous_rate):
    if raw_rate is None:
        return None

    if previous_rate is None:
        return raw_rate

    return (
        ACCELERATION_SMOOTHING_ALPHA * raw_rate
        + (1.0 - ACCELERATION_SMOOTHING_ALPHA) * previous_rate
    )


def get_track_color(speed_rate_mph_per_10_frames):
    if speed_rate_mph_per_10_frames is None:
        return (0, 255, 255)

    abs_rate = abs(speed_rate_mph_per_10_frames)

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

    if speed_rate_mph_per_10_frames > 0:
        return interpolate_bgr((0, 255, 255), (0, 255, 0), ratio)

    return interpolate_bgr((0, 255, 255), (0, 0, 255), ratio)


def mph_to_feet_per_second(speed_mph):
    return speed_mph * 5280.0 / 3600.0


def format_optional_float(value, digits=4, missing=""):
    if value is None:
        return missing
    return f"{value:.{digits}f}"


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def format_following_tag(following):
    if following is None:
        return "FD --"

    status = following.get("following_status")
    if status == "no_leader":
        return "FD none"
    if status == "no_lane":
        return "FD no lane"

    distance_ft = following.get("following_distance_ft")
    following_time_s = following["following_time_s"]
    if following_time_s is None and distance_ft is None:
        return "FD --"
    if following_time_s is None:
        return f"FD {distance_ft:.0f}ft"
    if distance_ft is None:
        return f"FD {following_time_s:.1f}s"

    return f"FD {following_time_s:.1f}s {distance_ft:.0f}ft"


def compact_vehicle_type_name(vehicle_type):
    return {
        "big_car": "big",
        "motorcycle": "moto",
    }.get(vehicle_type, vehicle_type)


def build_vehicle_tag_lines(item, following):
    vehicle_type = compact_vehicle_type_name(item["vehicle_type"])
    tracker_id = item["tracker_id"]

    if tracker_id is None:
        return [f"{vehicle_type} {item['confidence']:.2f}"]

    speed_mph = item["speed_mph"]
    speed_text = f"{speed_mph:.0f}mph" if speed_mph is not None else "--mph"
    lane_text = f"L{item['lane']}" if item["lane"] is not None else "L-"

    return [
        f"ID {tracker_id} {vehicle_type}",
        f"{speed_text} {lane_text}",
        format_following_tag(following),
    ]


def get_front_gap_pixels(follower_bbox, candidate_bbox):
    return follower_bbox[0] - candidate_bbox[2]


def get_rear_gap_pixels(subject_bbox, candidate_bbox):
    return candidate_bbox[0] - subject_bbox[2]


def get_box_width_pixels(bbox):
    return max(bbox[2] - bbox[0], 1.0)


def make_following_result(status, frame_left_clearance_px=None):
    return {
        "leader_tracker_id": "",
        "following_status": status,
        "following_gap_px": None,
        "following_gap_car_lengths": None,
        "following_distance_ft": None,
        "following_time_s": None,
        "front_clearance_to_frame_edge_px": frame_left_clearance_px,
    }


def find_nearest_front_vehicle(subject, current_tracks, target_lane):
    best_vehicle = None
    best_gap_px = None

    if target_lane is None:
        return None, None

    for candidate in current_tracks:
        if candidate["tracker_id"] == subject["tracker_id"]:
            continue
        if candidate["lane"] != target_lane:
            continue

        gap_px = get_front_gap_pixels(subject["bbox"], candidate["bbox"])
        if gap_px < 0:
            continue

        if best_gap_px is None or gap_px < best_gap_px:
            best_gap_px = gap_px
            best_vehicle = candidate

    return best_vehicle, best_gap_px


def find_nearest_rear_vehicle(subject, current_tracks, target_lane):
    best_vehicle = None
    best_gap_px = None

    if target_lane is None:
        return None, None

    for candidate in current_tracks:
        if candidate["tracker_id"] == subject["tracker_id"]:
            continue
        if candidate["lane"] != target_lane:
            continue

        gap_px = get_rear_gap_pixels(subject["bbox"], candidate["bbox"])
        if gap_px < 0:
            continue

        if best_gap_px is None or gap_px < best_gap_px:
            best_gap_px = gap_px
            best_vehicle = candidate

    return best_vehicle, best_gap_px


def compute_following_distances(current_tracks, feet_per_pixel):
    following_by_track = {}

    for follower in current_tracks:
        follower_lane = follower["lane"]
        follower_speed_mph = follower["speed_mph"]

        if follower_lane is None:
            following_by_track[follower["tracker_id"]] = make_following_result("no_lane")
            continue

        # Sweep left from the follower's front edge within the current lane.
        # The first back edge hit is the leader. If no edge is hit before the
        # frame boundary, this vehicle has no visible leader in its lane.
        best_leader, best_gap_px = find_nearest_front_vehicle(
            follower,
            current_tracks,
            follower_lane,
        )
        if best_leader is None:
            following_by_track[follower["tracker_id"]] = make_following_result(
                "no_leader",
                frame_left_clearance_px=follower["bbox"][0],
            )
            continue

        distance_ft = best_gap_px * feet_per_pixel
        if follower_speed_mph is None or follower_speed_mph <= 0:
            following_time_s = None
            status = "no_speed"
        else:
            speed_ft_per_second = mph_to_feet_per_second(follower_speed_mph)
            following_time_s = distance_ft / speed_ft_per_second
            status = "ok"

        following_by_track[follower["tracker_id"]] = {
            "leader_tracker_id": best_leader["tracker_id"],
            "following_status": status,
            "following_gap_px": best_gap_px,
            "following_gap_car_lengths": best_gap_px / get_box_width_pixels(follower["bbox"]),
            "following_distance_ft": distance_ft,
            "following_time_s": following_time_s,
            "front_clearance_to_frame_edge_px": None,
        }

    return following_by_track


def make_merge_result():
    return {
        "merge_event": False,
        "merge_from_lane": None,
        "merge_to_lane": None,
        "merge_front_vehicle_id": "",
        "merge_front_gap_px": None,
        "merge_front_gap_car_lengths": None,
        "merge_rear_vehicle_id": "",
        "merge_rear_gap_px": None,
        "merge_rear_gap_car_lengths": None,
        "merge_space_status": "",
        "merge_has_ample_space": "",
    }


def gap_is_ample(gap_car_lengths):
    return gap_car_lengths is None or gap_car_lengths >= AMPLE_MERGE_GAP_CAR_LENGTHS


def compute_merge_gaps(current_tracks):
    merge_by_track = {}

    for track in current_tracks:
        result = make_merge_result()
        merge_to_lane = track.get("merge_to_lane")
        if merge_to_lane is None:
            merge_by_track[track["tracker_id"]] = result
            continue

        front_vehicle, front_gap_px = find_nearest_front_vehicle(
            track,
            current_tracks,
            merge_to_lane,
        )
        rear_vehicle, rear_gap_px = find_nearest_rear_vehicle(
            track,
            current_tracks,
            merge_to_lane,
        )
        car_length_px = get_box_width_pixels(track["bbox"])
        front_gap_car_lengths = (
            front_gap_px / car_length_px if front_gap_px is not None else None
        )
        rear_gap_car_lengths = (
            rear_gap_px / car_length_px if rear_gap_px is not None else None
        )
        has_ample_space = (
            gap_is_ample(front_gap_car_lengths)
            and gap_is_ample(rear_gap_car_lengths)
        )

        result.update({
            "merge_event": True,
            "merge_from_lane": track.get("merge_from_lane"),
            "merge_to_lane": merge_to_lane,
            "merge_front_vehicle_id": (
                front_vehicle["tracker_id"] if front_vehicle is not None else ""
            ),
            "merge_front_gap_px": front_gap_px,
            "merge_front_gap_car_lengths": front_gap_car_lengths,
            "merge_rear_vehicle_id": (
                rear_vehicle["tracker_id"] if rear_vehicle is not None else ""
            ),
            "merge_rear_gap_px": rear_gap_px,
            "merge_rear_gap_car_lengths": rear_gap_car_lengths,
            "merge_space_status": "ample" if has_ample_space else "tight",
            "merge_has_ample_space": "yes" if has_ample_space else "no",
        })
        merge_by_track[track["tracker_id"]] = result

    return merge_by_track


def update_stable_lane(track_id, observed_lane, stable_lanes, pending_lanes, pending_counts):
    stable_lane = stable_lanes.get(track_id)
    if observed_lane is None:
        return stable_lane, None, None

    if stable_lane is None:
        stable_lanes[track_id] = observed_lane
        pending_lanes.pop(track_id, None)
        pending_counts.pop(track_id, None)
        return observed_lane, None, None

    if observed_lane == stable_lane:
        pending_lanes.pop(track_id, None)
        pending_counts.pop(track_id, None)
        return stable_lane, None, None

    if pending_lanes.get(track_id) == observed_lane:
        pending_counts[track_id] += 1
    else:
        pending_lanes[track_id] = observed_lane
        pending_counts[track_id] = 1

    if pending_counts[track_id] < MERGE_LANE_CONFIRMATION_FRAMES:
        return stable_lane, None, None

    merge_from_lane = stable_lane
    merge_to_lane = observed_lane
    stable_lanes[track_id] = observed_lane
    pending_lanes.pop(track_id, None)
    pending_counts.pop(track_id, None)
    return observed_lane, merge_from_lane, merge_to_lane


def get_lane_boundary_candidate(track):
    lane_contacts = track.get("lane_contacts", [])
    if len(lane_contacts) < 2:
        return None

    contact_lanes = {contact["lane"] for contact in lane_contacts}
    source_lane = track.get("previous_stable_lane")
    if source_lane not in contact_lanes:
        source_lane = track.get("stable_lane")
    if source_lane not in contact_lanes:
        source_lane = track.get("lane")
    if source_lane not in contact_lanes:
        source_lane = lane_contacts[0]["lane"]

    target_contact = None
    for contact in lane_contacts:
        if contact["lane"] == source_lane:
            continue
        if target_contact is None:
            target_contact = contact
            continue
        if abs(contact["lane"] - source_lane) < abs(target_contact["lane"] - source_lane):
            target_contact = contact
        elif (
            abs(contact["lane"] - source_lane) == abs(target_contact["lane"] - source_lane)
            and contact["overlap_ratio"] > target_contact["overlap_ratio"]
        ):
            target_contact = contact

    if target_contact is None:
        return None

    return {
        "from_lane": source_lane,
        "to_lane": target_contact["lane"],
    }


def get_target_lane_gap_snapshot(track, current_tracks, target_lane):
    front_vehicle, front_gap_px = find_nearest_front_vehicle(
        track,
        current_tracks,
        target_lane,
    )
    rear_vehicle, rear_gap_px = find_nearest_rear_vehicle(
        track,
        current_tracks,
        target_lane,
    )
    car_length_px = get_box_width_pixels(track["bbox"])
    return {
        "front_vehicle_id": front_vehicle["tracker_id"] if front_vehicle is not None else "",
        "front_gap_px": front_gap_px,
        "front_gap_car_lengths": (
            front_gap_px / car_length_px if front_gap_px is not None else None
        ),
        "rear_vehicle_id": rear_vehicle["tracker_id"] if rear_vehicle is not None else "",
        "rear_gap_px": rear_gap_px,
        "rear_gap_car_lengths": (
            rear_gap_px / car_length_px if rear_gap_px is not None else None
        ),
    }


def get_track_pixel_snapshot(track):
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = track["bbox"]
    return {
        "frame": track["frame"],
        "time_s": track["video_time_s"],
        "center_x_px": track["center_x_px"],
        "center_y_px": track["center_y_px"],
        "bbox_x1": bbox_x1,
        "bbox_y1": bbox_y1,
        "bbox_x2": bbox_x2,
        "bbox_y2": bbox_y2,
        "lane": track["lane"],
        "stable_lane": track["stable_lane"],
        "lane_contacts": track.get("lane_contacts", []),
    }


def make_lane_crossing_event(event_id, track, lane_boxes, current_tracks, from_lane, to_lane):
    gap_snapshot = get_target_lane_gap_snapshot(track, current_tracks, to_lane)
    start_snapshot = get_track_pixel_snapshot(track)
    return {
        "event_id": event_id,
        "tracker_id": track["tracker_id"],
        "vehicle_type": track["vehicle_type"],
        "from_lane": from_lane,
        "to_lane": to_lane,
        "boundary_y_px": get_lane_boundary_y_px(lane_boxes, from_lane, to_lane),
        "start": start_snapshot,
        "start_from_lane_overlap_ratio": get_lane_contact_ratio(
            track.get("lane_contacts", []),
            from_lane,
        ),
        "start_to_lane_overlap_ratio": get_lane_contact_ratio(
            track.get("lane_contacts", []),
            to_lane,
        ),
        "start_front_vehicle_id": gap_snapshot["front_vehicle_id"],
        "start_front_gap_px": gap_snapshot["front_gap_px"],
        "start_front_gap_car_lengths": gap_snapshot["front_gap_car_lengths"],
        "start_rear_vehicle_id": gap_snapshot["rear_vehicle_id"],
        "start_rear_gap_px": gap_snapshot["rear_gap_px"],
        "start_rear_gap_car_lengths": gap_snapshot["rear_gap_car_lengths"],
        "last": start_snapshot,
    }


def format_event_value(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def make_lane_crossing_event_row(event, end_track, current_tracks, end_reason):
    if end_track is None:
        end_snapshot = event["last"]
        end_lane_contacts = end_snapshot.get("lane_contacts", [])
        end_gap_snapshot = {
            "front_vehicle_id": "",
            "front_gap_px": None,
            "front_gap_car_lengths": None,
            "rear_vehicle_id": "",
            "rear_gap_px": None,
            "rear_gap_car_lengths": None,
        }
    else:
        end_snapshot = get_track_pixel_snapshot(end_track)
        end_lane_contacts = end_track.get("lane_contacts", [])
        end_gap_snapshot = get_target_lane_gap_snapshot(
            end_track,
            current_tracks,
            event["to_lane"],
        )

    start = event["start"]
    duration_frames = end_snapshot["frame"] - start["frame"] + 1
    duration_s = end_snapshot["time_s"] - start["time_s"]

    return {
        "event_id": event["event_id"],
        "tracker_id": event["tracker_id"],
        "vehicle_type": event["vehicle_type"],
        "from_lane": event["from_lane"],
        "to_lane": event["to_lane"],
        "boundary_y_px": format_event_value(event["boundary_y_px"]),
        "start_frame": start["frame"],
        "start_time_s": format_event_value(start["time_s"]),
        "start_center_x_px": format_event_value(start["center_x_px"], 2),
        "start_center_y_px": format_event_value(start["center_y_px"], 2),
        "start_bbox_x1": format_event_value(start["bbox_x1"], 2),
        "start_bbox_y1": format_event_value(start["bbox_y1"], 2),
        "start_bbox_x2": format_event_value(start["bbox_x2"], 2),
        "start_bbox_y2": format_event_value(start["bbox_y2"], 2),
        "start_from_lane_overlap_ratio": format_event_value(
            event["start_from_lane_overlap_ratio"],
        ),
        "start_to_lane_overlap_ratio": format_event_value(
            event["start_to_lane_overlap_ratio"],
        ),
        "start_front_vehicle_id": event["start_front_vehicle_id"],
        "start_front_gap_px": format_event_value(event["start_front_gap_px"], 2),
        "start_front_gap_car_lengths": format_event_value(
            event["start_front_gap_car_lengths"],
        ),
        "start_rear_vehicle_id": event["start_rear_vehicle_id"],
        "start_rear_gap_px": format_event_value(event["start_rear_gap_px"], 2),
        "start_rear_gap_car_lengths": format_event_value(
            event["start_rear_gap_car_lengths"],
        ),
        "end_frame": end_snapshot["frame"],
        "end_time_s": format_event_value(end_snapshot["time_s"]),
        "end_center_x_px": format_event_value(end_snapshot["center_x_px"], 2),
        "end_center_y_px": format_event_value(end_snapshot["center_y_px"], 2),
        "end_bbox_x1": format_event_value(end_snapshot["bbox_x1"], 2),
        "end_bbox_y1": format_event_value(end_snapshot["bbox_y1"], 2),
        "end_bbox_x2": format_event_value(end_snapshot["bbox_x2"], 2),
        "end_bbox_y2": format_event_value(end_snapshot["bbox_y2"], 2),
        "end_lane": format_event_value(end_snapshot["lane"]),
        "end_stable_lane": format_event_value(end_snapshot["stable_lane"]),
        "end_from_lane_overlap_ratio": format_event_value(
            get_lane_contact_ratio(end_lane_contacts, event["from_lane"]),
        ),
        "end_to_lane_overlap_ratio": format_event_value(
            get_lane_contact_ratio(end_lane_contacts, event["to_lane"]),
        ),
        "end_front_vehicle_id": end_gap_snapshot["front_vehicle_id"],
        "end_front_gap_px": format_event_value(end_gap_snapshot["front_gap_px"], 2),
        "end_front_gap_car_lengths": format_event_value(
            end_gap_snapshot["front_gap_car_lengths"],
        ),
        "end_rear_vehicle_id": end_gap_snapshot["rear_vehicle_id"],
        "end_rear_gap_px": format_event_value(end_gap_snapshot["rear_gap_px"], 2),
        "end_rear_gap_car_lengths": format_event_value(
            end_gap_snapshot["rear_gap_car_lengths"],
        ),
        "duration_frames": duration_frames,
        "duration_s": format_event_value(duration_s),
        "end_reason": end_reason,
    }


def update_lane_crossing_events(
    current_tracks,
    lane_boxes,
    active_lane_crossings,
    lane_crossing_rows,
    next_event_id,
):
    current_tracks_by_id = {track["tracker_id"]: track for track in current_tracks}
    boundary_info_by_track = {}

    for track in current_tracks:
        tracker_id = track["tracker_id"]
        candidate = get_lane_boundary_candidate(track)
        active_event = active_lane_crossings.get(tracker_id)
        contact_lanes = {contact["lane"] for contact in track.get("lane_contacts", [])}

        if active_event is not None:
            active_event["last"] = get_track_pixel_snapshot(track)
            is_touching_boundary = (
                active_event["from_lane"] in contact_lanes
                and active_event["to_lane"] in contact_lanes
            )
            if is_touching_boundary:
                boundary_info_by_track[tracker_id] = {
                    "event_id": active_event["event_id"],
                    "from_lane": active_event["from_lane"],
                    "to_lane": active_event["to_lane"],
                    "boundary_y_px": active_event["boundary_y_px"],
                }

            if active_event["from_lane"] not in contact_lanes:
                lane_crossing_rows.append(make_lane_crossing_event_row(
                    active_event,
                    track,
                    current_tracks,
                    "no_longer_touching_from_lane",
                ))
                active_lane_crossings.pop(tracker_id, None)
            elif active_event["to_lane"] not in contact_lanes:
                lane_crossing_rows.append(make_lane_crossing_event_row(
                    active_event,
                    track,
                    current_tracks,
                    "returned_to_from_lane",
                ))
                active_lane_crossings.pop(tracker_id, None)
            continue

        if candidate is None:
            continue

        active_event = make_lane_crossing_event(
            next_event_id,
            track,
            lane_boxes,
            current_tracks,
            candidate["from_lane"],
            candidate["to_lane"],
        )
        active_lane_crossings[tracker_id] = active_event
        boundary_info_by_track[tracker_id] = {
            "event_id": active_event["event_id"],
            "from_lane": active_event["from_lane"],
            "to_lane": active_event["to_lane"],
            "boundary_y_px": active_event["boundary_y_px"],
        }
        next_event_id += 1

    for tracker_id in list(active_lane_crossings):
        if tracker_id in current_tracks_by_id:
            continue
        active_event = active_lane_crossings.pop(tracker_id)
        lane_crossing_rows.append(make_lane_crossing_event_row(
            active_event,
            None,
            current_tracks,
            "track_disappeared",
        ))

    return boundary_info_by_track, next_event_id


def close_open_lane_crossing_events(active_lane_crossings, lane_crossing_rows):
    for tracker_id in list(active_lane_crossings):
        active_event = active_lane_crossings.pop(tracker_id)
        lane_crossing_rows.append(make_lane_crossing_event_row(
            active_event,
            None,
            [],
            "video_ended",
        ))


def update_track_frame_stats(stats, following, merge):
    if following is not None:
        following_distance_ft = following["following_distance_ft"]
        following_time_s = following["following_time_s"]
        if following_distance_ft is not None:
            stats["following_distance_samples_ft"].append(following_distance_ft)
        if following_time_s is not None:
            stats["following_time_samples_s"].append(following_time_s)

    if not merge["merge_event"]:
        return

    stats["merge_event_count"] += 1
    if merge["merge_has_ample_space"] == "yes":
        stats["ample_merge_event_count"] += 1
    else:
        stats["tight_merge_event_count"] += 1


def build_track_telemetry_row(track, following, merge, boundary_info, feet_per_pixel):
    following = following or make_following_result("not_computed")
    merge = merge or make_merge_result()
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = track["bbox"]

    return {
        "frame": track["frame"],
        "video_time_s": f"{track['video_time_s']:.4f}",
        "tracker_id": track["tracker_id"],
        "vehicle_type": track["vehicle_type"],
        "lane": track["lane"] if track["lane"] is not None else "",
        "center_x_px": f"{track['center_x_px']:.2f}",
        "center_y_px": f"{track['center_y_px']:.2f}",
        "center_x_ft": f"{(track['center_x_px'] * feet_per_pixel):.4f}",
        "center_y_ft": f"{(track['center_y_px'] * feet_per_pixel):.4f}",
        "bbox_x1": f"{bbox_x1:.2f}",
        "bbox_y1": f"{bbox_y1:.2f}",
        "bbox_x2": f"{bbox_x2:.2f}",
        "bbox_y2": f"{bbox_y2:.2f}",
        "speed_mph": format_optional_float(track["speed_mph"]),
        "acceleration_mph_per_10_frames": format_optional_float(
            track["acceleration_mph_per_10_frames"]
        ),
        "motion_direction": VEHICLE_TRAVEL_DIRECTION,
        "leader_tracker_id": following["leader_tracker_id"],
        "following_status": following["following_status"],
        "following_gap_px": format_optional_float(
            following["following_gap_px"],
            missing=NO_NUMERIC_VALUE,
        ),
        "following_gap_car_lengths": format_optional_float(
            following["following_gap_car_lengths"],
            missing=NO_NUMERIC_VALUE,
        ),
        "following_distance_ft": format_optional_float(
            following["following_distance_ft"],
            missing=NO_NUMERIC_VALUE,
        ),
        "following_time_s": format_optional_float(
            following["following_time_s"],
            missing=NO_NUMERIC_VALUE,
        ),
        "front_clearance_to_frame_edge_px": format_optional_float(
            following["front_clearance_to_frame_edge_px"],
            missing=NO_NUMERIC_VALUE,
        ),
        "stable_lane": track["stable_lane"] if track["stable_lane"] is not None else "",
        "merge_event": "yes" if merge["merge_event"] else "no",
        "merge_from_lane": merge["merge_from_lane"] if merge["merge_from_lane"] is not None else "",
        "merge_to_lane": merge["merge_to_lane"] if merge["merge_to_lane"] is not None else "",
        "merge_front_vehicle_id": merge["merge_front_vehicle_id"],
        "merge_front_gap_px": format_optional_float(
            merge["merge_front_gap_px"],
            missing=NO_NUMERIC_VALUE,
        ),
        "merge_front_gap_car_lengths": format_optional_float(
            merge["merge_front_gap_car_lengths"],
            missing=NO_NUMERIC_VALUE,
        ),
        "merge_rear_vehicle_id": merge["merge_rear_vehicle_id"],
        "merge_rear_gap_px": format_optional_float(
            merge["merge_rear_gap_px"],
            missing=NO_NUMERIC_VALUE,
        ),
        "merge_rear_gap_car_lengths": format_optional_float(
            merge["merge_rear_gap_car_lengths"],
            missing=NO_NUMERIC_VALUE,
        ),
        "merge_space_status": merge["merge_space_status"],
        "merge_has_ample_space": merge["merge_has_ample_space"],
        "lane_boundary_contact": "yes" if boundary_info is not None else "no",
        "lane_boundary_event_id": boundary_info["event_id"] if boundary_info is not None else "",
        "lane_boundary_from_lane": boundary_info["from_lane"] if boundary_info is not None else "",
        "lane_boundary_to_lane": boundary_info["to_lane"] if boundary_info is not None else "",
        "lane_boundary_y_px": (
            format_optional_float(boundary_info["boundary_y_px"], missing=NO_NUMERIC_VALUE)
            if boundary_info is not None else NO_NUMERIC_VALUE
        ),
        "lane_contact_lanes": format_lane_contact_lanes(track["lane_contacts"]),
    }


def measure_vehicle_tag(lines):
    metrics = []
    max_width = 0
    total_text_height = 0

    for line in lines:
        text_size, baseline = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            VEHICLE_TAG_FONT_SCALE,
            VEHICLE_TAG_FONT_THICKNESS,
        )
        text_width, text_height = text_size
        metrics.append((text_width, text_height, baseline))
        max_width = max(max_width, text_width)
        total_text_height += text_height + baseline

    tag_width = max_width + (VEHICLE_TAG_PADDING_X * 2)
    tag_height = (
        total_text_height
        + (VEHICLE_TAG_LINE_GAP * max(len(lines) - 1, 0))
        + (VEHICLE_TAG_PADDING_Y * 2)
    )
    return tag_width, tag_height, metrics


def expand_rect(rect, pixels):
    x1, y1, x2, y2 = rect
    return (
        x1 - pixels,
        y1 - pixels,
        x2 + pixels,
        y2 + pixels,
    )


def choose_vehicle_tag_rect(frame_shape, bbox, tag_width, tag_height, placed_rects):
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    box_center_x = int(round((x1 + x2) / 2.0))
    box_center_y = int(round((y1 + y2) / 2.0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    bases = [
        (box_center_x - tag_width // 2, y1 - tag_height - VEHICLE_TAG_GAP),
        (box_center_x - tag_width // 2, y2 + VEHICLE_TAG_GAP),
        (x1, y1 - tag_height - VEHICLE_TAG_GAP),
        (x1, y2 + VEHICLE_TAG_GAP),
        (x2 - tag_width, y1 - tag_height - VEHICLE_TAG_GAP),
        (x2 - tag_width, y2 + VEHICLE_TAG_GAP),
        (box_center_x - tag_width // 2, y1 + VEHICLE_TAG_GAP),
    ]
    x_step = max(tag_width // 2, 24)
    y_step = tag_height + VEHICLE_TAG_GAP
    x_offsets = [0, -x_step, x_step, -tag_width, tag_width]
    y_offsets = [0, -y_step, y_step, -(2 * y_step), 2 * y_step]

    best_rect = None
    best_score = None
    seen = set()

    for base_index, (base_x, base_y) in enumerate(bases):
        for y_offset in y_offsets:
            for x_offset in x_offsets:
                tag_x = int(clamp(base_x + x_offset, 0, max(frame_w - tag_width - 1, 0)))
                tag_y = int(clamp(base_y + y_offset, 0, max(frame_h - tag_height - 1, 0)))
                rect = (tag_x, tag_y, tag_x + tag_width, tag_y + tag_height)
                if rect in seen:
                    continue
                seen.add(rect)

                expanded = expand_rect(rect, VEHICLE_TAG_MIN_GAP)
                overlap = sum(intersect_area(expanded, placed) for placed in placed_rects)
                rect_center_x = tag_x + (tag_width / 2.0)
                rect_center_y = tag_y + (tag_height / 2.0)
                distance = abs(rect_center_x - box_center_x) + abs(rect_center_y - box_center_y)
                score = (overlap * 1000.0) + distance + (base_index * 5.0)

                if overlap <= 0:
                    return rect

                if best_score is None or score < best_score:
                    best_score = score
                    best_rect = rect

    return best_rect


def blend_rect(frame, rect, color, alpha):
    x1, y1, x2, y2 = rect
    x1 = int(clamp(x1, 0, frame.shape[1] - 1))
    y1 = int(clamp(y1, 0, frame.shape[0] - 1))
    x2 = int(clamp(x2, x1 + 1, frame.shape[1]))
    y2 = int(clamp(y2, y1 + 1, frame.shape[0]))

    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def draw_vehicle_tag(frame, lines, bbox, placed_rects):
    tag_width, tag_height, metrics = measure_vehicle_tag(lines)
    rect = choose_vehicle_tag_rect(frame.shape, bbox, tag_width, tag_height, placed_rects)
    if rect is None:
        return

    x1, y1, x2, y2 = rect
    box_x1, box_y1, box_x2, box_y2 = bbox
    box_center_x = int(round((box_x1 + box_x2) / 2.0))
    box_top_y = int(round(box_y1))
    box_bottom_y = int(round(box_y2))
    tag_anchor_x = int(clamp(box_center_x, x1, x2))
    if y1 < box_top_y:
        tag_anchor_y = y2
        box_anchor_y = box_top_y
    else:
        tag_anchor_y = y1
        box_anchor_y = box_bottom_y

    cv2.line(
        frame,
        (tag_anchor_x, tag_anchor_y),
        (box_center_x, box_anchor_y),
        VEHICLE_TAG_LEADER_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    blend_rect(frame, rect, VEHICLE_TAG_BG_COLOR, VEHICLE_TAG_BG_ALPHA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_TAG_BORDER_COLOR, 1, cv2.LINE_AA)

    text_y = y1 + VEHICLE_TAG_PADDING_Y
    for index, (line, (_, text_height, baseline)) in enumerate(zip(lines, metrics)):
        text_y += text_height
        text_color = VEHICLE_TAG_TEXT_COLOR if index == 0 else VEHICLE_TAG_MUTED_TEXT_COLOR
        cv2.putText(
            frame,
            line,
            (x1 + VEHICLE_TAG_PADDING_X, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            VEHICLE_TAG_FONT_SCALE,
            text_color,
            VEHICLE_TAG_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        text_y += baseline + VEHICLE_TAG_LINE_GAP

    placed_rects.append(expand_rect(rect, VEHICLE_TAG_MIN_GAP))


def draw_overlay(frame, frame_num, frame_time, raw_dets, active_tracks, total_unique, avg_speed_mph):
    avg_speed_text = f"{avg_speed_mph:.1f} mph" if avg_speed_mph is not None else "--"
    lines = [
        f"Frame: {frame_num}",
        f"Time: {frame_time:.4f}s",
        f"Raw Detections: {raw_dets}",
        f"Active Tracks: {active_tracks}",
        f"Unique Vehicles: {total_unique}",
        f"Avg Active Speed: {avg_speed_text}",
    ]

    for row, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 35 + (row * 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )


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


def iter_vehicle_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return

    ids = boxes.id.int().cpu().tolist() if boxes.id is not None else None
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.int().cpu().tolist()

    for index, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.astype(int)
        track_id = int(ids[index]) if ids is not None and index < len(ids) else None
        yield {
            "bbox": (x1, y1, x2, y2),
            "confidence": float(confidences[index]),
            "vehicle_type": get_vehicle_class_name(class_ids[index]),
            "tracker_id": track_id,
        }


def draw_vehicle_items(frame, vehicle_draw_items, following_by_track):
    placed_tag_rects = []
    ordered_items = sorted(
        vehicle_draw_items,
        key=lambda item: (item["bbox"][1], item["bbox"][0]),
    )

    for item in ordered_items:
        x1, y1, x2, y2 = item["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), item["color"], LINE_WIDTH)
        following = following_by_track.get(item["tracker_id"]) if item["tracker_id"] is not None else None
        tag_lines = build_vehicle_tag_lines(item, following)
        draw_vehicle_tag(frame, tag_lines, item["bbox"], placed_tag_rects)


def write_report_files(
    video_path,
    fps,
    frame_num,
    lane_boxes,
    track_stats,
    track_speed_samples,
    telemetry_rows,
    lane_crossing_rows,
    metrics_csv_path,
    telemetry_csv_path,
    merge_events_csv_path,
    speed_report_path,
):
    write_speed_report(speed_report_path, video_path, fps, frame_num, track_speed_samples)
    metrics_csv_path, metrics_redirected = write_vehicle_metrics_csv(
        metrics_csv_path,
        fps,
        lane_boxes,
        track_stats,
    )
    telemetry_csv_path, telemetry_redirected = write_vehicle_telemetry_csv(
        telemetry_csv_path,
        telemetry_rows,
    )
    merge_events_csv_path, merge_redirected = write_merge_events_csv(
        merge_events_csv_path,
        lane_crossing_rows,
    )
    return (
        metrics_csv_path,
        telemetry_csv_path,
        merge_events_csv_path,
        metrics_redirected,
        telemetry_redirected,
        merge_redirected,
    )


def print_redirect_notice(label, csv_path, redirected):
    if redirected:
        print(
            f"{label} CSV target was locked. Writing to fallback path: {csv_path}",
            flush=True,
        )


def make_speed_state():
    return {
        "histories": defaultdict(lambda: deque(maxlen=SPEED_ESTIMATE_HISTORY_POINTS)),
        "speeds_mph": {},
        "motion_slopes": {},
        "rates_mph_per_10_frames": {},
        "non_edge_frames": defaultdict(int),
        "raw_histories": defaultdict(lambda: deque(maxlen=RAW_SPEED_HISTORY_SIZE)),
        "rate_histories": defaultdict(lambda: deque(maxlen=ACCELERATION_HISTORY_SIZE)),
        "samples": defaultdict(list),
    }


def reset_track_speed_state(track_id, speed_state):
    speed_state["non_edge_frames"][track_id] = 0
    speed_state["histories"][track_id].clear()
    speed_state["raw_histories"][track_id].clear()
    speed_state["rate_histories"][track_id].clear()
    speed_state["motion_slopes"].pop(track_id, None)
    speed_state["rates_mph_per_10_frames"].pop(track_id, None)


def update_track_speed(
    track_id,
    frame_num,
    current_time_s,
    center_x,
    center_y,
    bbox,
    frame_size,
    fps,
    feet_per_pixel,
    speed_state,
    stats,
):
    speed_mph = None
    previous_speed_mph = speed_state["speeds_mph"].get(track_id)
    frame_width, frame_height = frame_size

    if is_box_near_frame_edge(bbox, frame_width, frame_height, EDGE_MARGIN_PIXELS):
        reset_track_speed_state(track_id, speed_state)
    else:
        speed_state["non_edge_frames"][track_id] += 1
        history = speed_state["histories"][track_id]

        if speed_state["non_edge_frames"][track_id] >= NON_EDGE_WARMUP_FRAMES:
            if not history or frame_num - history[-1][0] >= SPEED_ESTIMATE_FRAME_GAP:
                history.append((frame_num, (center_x, center_y)))

            motion_slope = estimate_motion_slope_pixels_per_frame(history)
            if motion_slope is not None:
                speed_state["motion_slopes"][track_id] = motion_slope

            raw_speed_mph = estimate_speed_mph(history, fps, feet_per_pixel)
            if raw_speed_mph is not None:
                speed_state["raw_histories"][track_id].append(raw_speed_mph)
                speed_mph = smooth_speed_mph(
                    speed_state["raw_histories"][track_id],
                    previous_speed_mph,
                )

    if speed_mph is not None:
        speed_state["speeds_mph"][track_id] = speed_mph
        speed_state["samples"][track_id].append({
            "frame": frame_num,
            "video_time_s": current_time_s,
            "speed_mph": speed_mph,
        })
        stats["speed_samples_mph"].append(speed_mph)

        rate_history = speed_state["rate_histories"][track_id]
        rate_history.append((frame_num, speed_mph))
        raw_rate = estimate_speed_rate_mph_per_10_frames(rate_history)
        smoothed_rate = smooth_speed_rate_mph_per_10_frames(
            raw_rate,
            speed_state["rates_mph_per_10_frames"].get(track_id),
        )
        if smoothed_rate is not None:
            speed_state["rates_mph_per_10_frames"][track_id] = smoothed_rate
            stats["acceleration_samples_mph_per_10_frames"].append(smoothed_rate)

    return (
        speed_state["speeds_mph"].get(track_id),
        speed_state["rates_mph_per_10_frames"].get(track_id),
        speed_state["motion_slopes"].get(track_id),
    )


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

    output_paths = make_output_paths(video_path)
    out_path = output_paths["video"]
    speed_report_path = output_paths["speed_report"]
    metrics_csv_path = output_paths["metrics_csv"]
    telemetry_csv_path = output_paths["telemetry_csv"]
    merge_events_csv_path = output_paths["merge_events_csv"]
    following_graph_path = output_paths["following_graph"]
    lane_boxes = load_lane_boxes(LANE_ANNOTATION_XML)
    metrics_path_was_redirected = False
    telemetry_path_was_redirected = False
    merge_events_path_was_redirected = False

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
    speed_state = make_speed_state()
    track_speed_samples = speed_state["samples"]
    telemetry_rows = []
    lane_crossing_rows = []
    active_lane_crossings = {}
    next_lane_crossing_event_id = 1
    track_stats = defaultdict(initialize_track_stats)
    track_stable_lanes = {}
    track_pending_lanes = {}
    track_pending_lane_counts = {}
    frame_num = 0
    total_time = 0.0
    stop_requested = False
    # Horizontal calibration: the full visible width is estimated from 10 ft lane dashes.
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
            current_frame_tracks = []
            vehicle_draw_items = []

            if result.boxes is not None and len(result.boxes) > 0:
                detections = list(iter_vehicle_detections(result))
                raw_dets = len(detections)

                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    conf = detection["confidence"]
                    class_name = detection["vehicle_type"]
                    track_id = detection["tracker_id"]

                    # count active tracks only when id exists
                    if track_id is not None:
                        active_tracks += 1
                        track_id = int(track_id)
                        seen_ids.add(track_id)

                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0

                        current_time_s = frame_num / fps
                        stats = track_stats[track_id]
                        if stats["vehicle_type"] is None:
                            stats["vehicle_type"] = class_name
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
                        lane_contacts = get_lane_contacts(
                            (float(x1), float(y1), float(x2), float(y2)),
                            lane_boxes,
                        )
                        previous_stable_lane = track_stable_lanes.get(track_id)
                        if majority_lane is not None:
                            stats["lane_frames"][majority_lane] += 1
                        stable_lane, merge_from_lane, merge_to_lane = update_stable_lane(
                            track_id,
                            majority_lane,
                            track_stable_lanes,
                            track_pending_lanes,
                            track_pending_lane_counts,
                        )

                        speed_mph, acceleration_mph_per_10_frames, motion_slope = update_track_speed(
                            track_id,
                            frame_num,
                            current_time_s,
                            center_x,
                            center_y,
                            (x1, y1, x2, y2),
                            (w, h),
                            fps,
                            feet_per_pixel,
                            speed_state,
                            stats,
                        )
                        if speed_mph is not None:
                            active_speed_samples.append(speed_mph)
                        color = get_track_color(
                            acceleration_mph_per_10_frames,
                        )
                        current_frame_tracks.append({
                            "frame": frame_num,
                            "video_time_s": current_time_s,
                            "tracker_id": track_id,
                            "vehicle_type": class_name,
                            "lane": majority_lane,
                            "center_x_px": center_x,
                            "center_y_px": center_y,
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                            "speed_mph": speed_mph,
                            "acceleration_mph_per_10_frames": acceleration_mph_per_10_frames,
                            "motion_slope_px_per_frame": motion_slope,
                            "previous_stable_lane": previous_stable_lane,
                            "stable_lane": stable_lane,
                            "lane_contacts": lane_contacts,
                            "merge_from_lane": merge_from_lane,
                            "merge_to_lane": merge_to_lane,
                        })
                    else:
                        speed_mph = None
                        majority_lane = None
                        color = (0, 165, 255)

                    vehicle_draw_items.append({
                        "tracker_id": track_id,
                        "vehicle_type": class_name,
                        "confidence": float(conf),
                        "speed_mph": speed_mph,
                        "lane": majority_lane,
                        "bbox": (x1, y1, x2, y2),
                        "color": color,
                    })

            following_by_track = compute_following_distances(current_frame_tracks, feet_per_pixel)
            merge_by_track = compute_merge_gaps(current_frame_tracks)
            boundary_info_by_track, next_lane_crossing_event_id = update_lane_crossing_events(
                current_frame_tracks,
                lane_boxes,
                active_lane_crossings,
                lane_crossing_rows,
                next_lane_crossing_event_id,
            )
            draw_vehicle_items(plotted, vehicle_draw_items, following_by_track)

            for track in current_frame_tracks:
                following = following_by_track.get(track["tracker_id"])
                merge = merge_by_track.get(track["tracker_id"], make_merge_result())
                boundary_info = boundary_info_by_track.get(track["tracker_id"])
                stats = track_stats[track["tracker_id"]]

                update_track_frame_stats(stats, following, merge)
                telemetry_rows.append(build_track_telemetry_row(
                    track,
                    following,
                    merge,
                    boundary_info,
                    feet_per_pixel,
                ))

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
                (
                    metrics_csv_path,
                    telemetry_csv_path,
                    merge_events_csv_path,
                    metrics_path_was_redirected,
                    telemetry_path_was_redirected,
                    merge_events_path_was_redirected,
                ) = write_report_files(
                    video_path,
                    fps,
                    frame_num,
                    lane_boxes,
                    track_stats,
                    track_speed_samples,
                    telemetry_rows,
                    lane_crossing_rows,
                    metrics_csv_path,
                    telemetry_csv_path,
                    merge_events_csv_path,
                    speed_report_path,
                )
                if frame_num == 1:
                    print_redirect_notice("Metrics", metrics_csv_path, metrics_path_was_redirected)
                    print_redirect_notice("Telemetry", telemetry_csv_path, telemetry_path_was_redirected)
                    print_redirect_notice(
                        "Merge events",
                        merge_events_csv_path,
                        merge_events_path_was_redirected,
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
        close_open_lane_crossing_events(active_lane_crossings, lane_crossing_rows)
        (
            metrics_csv_path,
            telemetry_csv_path,
            merge_events_csv_path,
            metrics_path_was_redirected,
            telemetry_path_was_redirected,
            merge_events_path_was_redirected,
        ) = write_report_files(
            video_path,
            fps,
            frame_num,
            lane_boxes,
            track_stats,
            track_speed_samples,
            telemetry_rows,
            lane_crossing_rows,
            metrics_csv_path,
            telemetry_csv_path,
            merge_events_csv_path,
            speed_report_path,
        )
        graph_written = write_following_distance_graph(
            following_graph_path,
            telemetry_rows,
        )

    avg_time = total_time / frame_num if frame_num > 0 else 0.0

    status_text = "Stopped early" if stop_requested else "Finished"
    print(f"\n{status_text}: {os.path.basename(video_path)}")
    print(f"Frames: {frame_num}")
    print(f"Avg frame time: {avg_time:.6f} s")
    print(f"Total vehicles: {len(seen_ids)}")
    print(f"Saved to: {out_path}\n")
    print(f"Speed report: {speed_report_path}")
    print(f"Vehicle metrics CSV: {metrics_csv_path}")
    print(f"Vehicle telemetry CSV: {telemetry_csv_path}")
    print(f"Merge events CSV: {merge_events_csv_path}")
    if graph_written:
        print(f"Following distance graph: {following_graph_path}\n")
    else:
        print("Following distance graph: no usable following-distance samples\n")
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
