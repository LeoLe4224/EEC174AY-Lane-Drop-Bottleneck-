import os
import cv2
import time
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "out_videos"

MODEL_WEIGHTS = "runs_highway/car_detector_ft/weights/best.pt"

# custom model has one class only: car
VEHICLE_CLASSES = [0]

CONF_THRESH = 0.03
IMG_SIZE = 1920
LINE_WIDTH = 2

SHOW_VIDEO = True

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


def draw_overlay(frame, frame_num, frame_time, raw_dets, active_tracks, total_unique):
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
    frame_num = 0
    total_time = 0.0

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
                    seen_ids.add(int(track_id))
                    label = f"ID {track_id} car {conf:.2f}"
                    color = (0, 255, 0)
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

        draw_overlay(plotted, frame_num, frame_time, raw_dets, active_tracks, len(seen_ids))

        print(
            f"{os.path.basename(video_path)} | "
            f"frame {frame_num} | "
            f"time {frame_time:.6f} s | "
            f"raw {raw_dets} | "
            f"active {active_tracks} | "
            f"total {len(seen_ids)}"
        )

        out_writer.write(plotted)

        if SHOW_VIDEO:
            cv2.imshow("Tracking", plotted)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    out_writer.release()

    avg_time = total_time / frame_num if frame_num > 0 else 0.0

    print(f"\nFinished: {os.path.basename(video_path)}")
    print(f"Frames: {frame_num}")
    print(f"Avg frame time: {avg_time:.6f} s")
    print(f"Total vehicles: {len(seen_ids)}")
    print(f"Saved to: {out_path}\n")


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
        process_video(model, video_path)

    if SHOW_VIDEO:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
