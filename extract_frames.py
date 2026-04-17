import os
import cv2

# =========================================================
# SETTINGS
# =========================================================

INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "traffic_dataset/images/all"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")

# save 1 frame every N frames
FRAME_EVERY = 15


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


def main():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Missing input folder: {INPUT_FOLDER}")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    videos = collect_videos(INPUT_FOLDER)

    if not videos:
        print("No videos found in input_videos/")
        return

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        base = os.path.splitext(os.path.basename(video_path))[0]

        frame_idx = 0
        save_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_EVERY == 0:
                out_name = f"{base}_frame_{save_idx:05d}.jpg"
                out_path = os.path.join(OUTPUT_FOLDER, out_name)
                cv2.imwrite(out_path, frame)

                print(f"Saved: {out_name}")
                save_idx += 1

            frame_idx += 1

        cap.release()
        print(f"Finished extracting from: {os.path.basename(video_path)}\n")


if __name__ == "__main__":
    main()