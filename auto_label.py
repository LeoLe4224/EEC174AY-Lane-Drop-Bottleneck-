import os
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

IMAGES_FOLDER = "traffic_dataset/images/all"
LABELS_FOLDER = "traffic_dataset/labels/all"

MODEL_WEIGHTS = "yolo11l.pt"

# COCO class id for car
COCO_CAR_CLASS = 2

# use very low confidence for auto-labeling
# better to get extra boxes and delete them later
AUTO_LABEL_CONF = 0.01

# larger image size helps small overhead cars
IMG_SIZE = 1920

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================
# HELPERS
# =========================================================

def collect_images(folder):
    imgs = []

    if not os.path.isdir(folder):
        return imgs

    for f in os.listdir(folder):
        fullpath = os.path.join(folder, f)

        if os.path.isfile(fullpath) and f.lower().endswith(IMG_EXTS):
            imgs.append(fullpath)

    return sorted(imgs)


def main():
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"Missing images folder: {IMAGES_FOLDER}")
        return

    os.makedirs(LABELS_FOLDER, exist_ok=True)

    images = collect_images(IMAGES_FOLDER)

    if not images:
        print("No images found.")
        return

    model = YOLO(MODEL_WEIGHTS)

    for img_path in images:
        results = model.predict(
            source=img_path,
            conf=AUTO_LABEL_CONF,
            imgsz=IMG_SIZE,
            classes=[COCO_CAR_CLASS],
            verbose=False
        )

        result = results[0]

        base = os.path.basename(img_path)
        stem = os.path.splitext(base)[0]
        label_path = os.path.join(LABELS_FOLDER, f"{stem}.txt")

        with open(label_path, "w", encoding="utf-8") as f:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xywhn = result.boxes.xywhn.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, cls_id, conf in zip(boxes_xywhn, classes, confs):
                    if int(cls_id) != COCO_CAR_CLASS:
                        continue

                    x, y, w, h = box.tolist()

                    # output class 0 for your custom dataset
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

                print(f"Labeled: {base} | boxes: {len(boxes_xywhn)}")
            else:
                print(f"Labeled: {base} | boxes: 0")

    print("\nDone auto-labeling.")
    print(f"Labels saved in: {LABELS_FOLDER}")


if __name__ == "__main__":
    main()