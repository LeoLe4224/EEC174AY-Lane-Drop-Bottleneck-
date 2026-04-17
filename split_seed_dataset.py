import os
import random
import shutil

# =========================================================
# SETTINGS
# =========================================================

SEED_IMAGE_FOLDER = "traffic_dataset/images/seed"
SEED_LABEL_FOLDER = "traffic_dataset/labels/seed"

TRAIN_IMAGE_FOLDER = "traffic_dataset/images/train"
VAL_IMAGE_FOLDER = "traffic_dataset/images/val"
TRAIN_LABEL_FOLDER = "traffic_dataset/labels/train"
VAL_LABEL_FOLDER = "traffic_dataset/labels/val"

DATA_YAML_PATH = "traffic_dataset/data.yaml"

VAL_RATIO = 0.2
RANDOM_SEED = 42

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)

        if os.path.isfile(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)


def write_data_yaml():
    yaml_text = """path: traffic_dataset
train: images/train
val: images/val

names:
  0: car
"""

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_text)


def convert_label_to_car_only(src_path, dst_path):
    # Rewrite every label row so the class id is always 0 for car.
    cleaned_lines = []

    with open(src_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            cleaned_line = f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}"
            cleaned_lines.append(cleaned_line)

    with open(dst_path, "w", encoding="utf-8") as f:
        if cleaned_lines:
            f.write("\n".join(cleaned_lines) + "\n")


def main():
    if not os.path.isdir(SEED_IMAGE_FOLDER):
        print(f"Missing image folder: {SEED_IMAGE_FOLDER}")
        return

    if not os.path.isdir(SEED_LABEL_FOLDER):
        print(f"Missing label folder: {SEED_LABEL_FOLDER}")
        return

    clear_folder(TRAIN_IMAGE_FOLDER)
    clear_folder(VAL_IMAGE_FOLDER)
    clear_folder(TRAIN_LABEL_FOLDER)
    clear_folder(VAL_LABEL_FOLDER)

    paired_files = []

    for image_name in sorted(os.listdir(SEED_IMAGE_FOLDER)):
        image_path = os.path.join(SEED_IMAGE_FOLDER, image_name)

        if not os.path.isfile(image_path):
            continue

        if not image_name.lower().endswith(IMAGE_EXTS):
            continue

        stem, _ = os.path.splitext(image_name)
        label_name = stem + ".txt"
        label_path = os.path.join(SEED_LABEL_FOLDER, label_name)

        if os.path.isfile(label_path):
            paired_files.append((image_name, label_name))

    if not paired_files:
        print("No labeled seed images found.")
        return

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(paired_files)

    val_count = max(1, int(len(paired_files) * VAL_RATIO)) if len(paired_files) > 1 else 0
    val_pairs = paired_files[:val_count]
    train_pairs = paired_files[val_count:]

    if not train_pairs and val_pairs:
        train_pairs.append(val_pairs.pop())

    for image_name, label_name in train_pairs:
        shutil.copy2(
            os.path.join(SEED_IMAGE_FOLDER, image_name),
            os.path.join(TRAIN_IMAGE_FOLDER, image_name),
        )
        convert_label_to_car_only(
            os.path.join(SEED_LABEL_FOLDER, label_name),
            os.path.join(TRAIN_LABEL_FOLDER, label_name),
        )

    for image_name, label_name in val_pairs:
        shutil.copy2(
            os.path.join(SEED_IMAGE_FOLDER, image_name),
            os.path.join(VAL_IMAGE_FOLDER, image_name),
        )
        convert_label_to_car_only(
            os.path.join(SEED_LABEL_FOLDER, label_name),
            os.path.join(VAL_LABEL_FOLDER, label_name),
        )

    write_data_yaml()

    print(f"Total labeled seed images: {len(paired_files)}")
    print(f"Train images: {len(train_pairs)}")
    print(f"Val images: {len(val_pairs)}")
    print(f"Saved: {DATA_YAML_PATH}")


if __name__ == "__main__":
    main()
