import os
import shutil

# =========================================================
# SETTINGS
# =========================================================

SOURCE_FOLDER = "traffic_dataset/images/all"
SEED_FOLDER = "traffic_dataset/images/seed"

# copy every Nth image from images/all into images/seed
EVERY_NTH = 10

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    if not os.path.isdir(SOURCE_FOLDER):
        print(f"Missing source folder: {SOURCE_FOLDER}")
        return

    os.makedirs(SEED_FOLDER, exist_ok=True)

    image_names = []
    for name in sorted(os.listdir(SOURCE_FOLDER)):
        full_path = os.path.join(SOURCE_FOLDER, name)
        if os.path.isfile(full_path) and name.lower().endswith(IMAGE_EXTS):
            image_names.append(name)

    if not image_names:
        print("No images found in traffic_dataset/images/all/")
        return

    copied_count = 0

    for index, image_name in enumerate(image_names):
        if index % EVERY_NTH != 0:
            continue

        src_path = os.path.join(SOURCE_FOLDER, image_name)
        dst_path = os.path.join(SEED_FOLDER, image_name)
        shutil.copy2(src_path, dst_path)
        copied_count += 1

    print(f"Copied {copied_count} images to {SEED_FOLDER}")


if __name__ == "__main__":
    main()
