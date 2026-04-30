import argparse
import shutil
from pathlib import Path

import cv2


RAW_ID_TO_CLASS_NAME = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "big_car",
    7: "tricycle",
    8: "awning-tricycle",
    9: "big_car",
    10: "motor",
}

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "big_car",
    "tricycle",
    "awning-tricycle",
    "motor",
]

CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw VisDrone-DET annotations to YOLO format and merge truck/bus into big_car."
    )
    parser.add_argument(
        "--train-root",
        default="datasets/visdrone_raw/VisDrone2019-DET-train",
        help="Raw VisDrone train directory containing images/ and annotations/.",
    )
    parser.add_argument(
        "--val-root",
        default="datasets/visdrone_raw/VisDrone2019-DET-val",
        help="Raw VisDrone val directory containing images/ and annotations/.",
    )
    parser.add_argument(
        "--output-root",
        default="datasets/visdrone",
        help="Output directory for YOLO-formatted images/ and labels/.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the output dataset. Defaults to hardlink, with copy fallback.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete generated train/val images and labels before rebuilding.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def prepare_output_dirs(output_root: Path, clear_output: bool):
    targets = [
        output_root / "images" / "train",
        output_root / "images" / "val",
        output_root / "labels" / "train",
        output_root / "labels" / "val",
    ]

    for target in targets:
        if clear_output:
            ensure_clean_dir(target)
        else:
            target.mkdir(parents=True, exist_ok=True)


def iter_annotation_files(annotations_dir: Path):
    return sorted(path for path in annotations_dir.glob("*.txt") if path.is_file())


def read_image_size(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid image size: {image_path}")

    return width, height


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def convert_annotation_file(annotation_path: Path, image_path: Path, output_label_path: Path):
    width, height = read_image_size(image_path)
    converted_lines = []

    for raw_line in annotation_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue

        bbox_left, bbox_top, bbox_width, bbox_height, score, category, _truncation, _occlusion = parts

        if int(score) == 0:
            continue

        raw_category = int(category)
        class_name = RAW_ID_TO_CLASS_NAME.get(raw_category)
        if class_name is None:
            continue

        x = float(bbox_left)
        y = float(bbox_top)
        w = float(bbox_width)
        h = float(bbox_height)

        if w <= 0 or h <= 0:
            continue

        x_center = clamp((x + w / 2.0) / width)
        y_center = clamp((y + h / 2.0) / height)
        norm_w = clamp(w / width)
        norm_h = clamp(h / height)

        if norm_w == 0.0 or norm_h == 0.0:
            continue

        class_index = CLASS_NAME_TO_INDEX[class_name]
        converted_lines.append(
            f"{class_index} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        )

    output_label_path.write_text(
        ("\n".join(converted_lines) + "\n") if converted_lines else "",
        encoding="utf-8",
    )


def materialize_image(src_image_path: Path, dst_image_path: Path, copy_images: bool):
    if dst_image_path.exists():
        return

    if copy_images:
        shutil.copy2(src_image_path, dst_image_path)
        return

    try:
        dst_image_path.hardlink_to(src_image_path)
    except OSError:
        shutil.copy2(src_image_path, dst_image_path)


def convert_split(split_name: str, split_root: Path, output_root: Path, copy_images: bool):
    images_dir = split_root / "images"
    annotations_dir = split_root / "annotations"

    if not images_dir.is_dir():
        raise RuntimeError(f"Missing images directory: {images_dir}")
    if not annotations_dir.is_dir():
        raise RuntimeError(f"Missing annotations directory: {annotations_dir}")

    output_images_dir = output_root / "images" / split_name
    output_labels_dir = output_root / "labels" / split_name

    converted = 0
    for annotation_path in iter_annotation_files(annotations_dir):
        image_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{annotation_path.stem}{ext}"
            if candidate.is_file():
                image_path = candidate
                break

        if image_path is None:
            raise RuntimeError(f"Missing image for annotation: {annotation_path.name}")

        dst_image_path = output_images_dir / image_path.name
        dst_label_path = output_labels_dir / f"{annotation_path.stem}.txt"

        materialize_image(image_path, dst_image_path, copy_images)
        convert_annotation_file(annotation_path, image_path, dst_label_path)
        converted += 1

    print(f"{split_name}: converted {converted} annotation files")


def main():
    args = parse_args()
    output_root = Path(args.output_root)

    prepare_output_dirs(output_root, clear_output=args.clear_output)
    convert_split("train", Path(args.train_root), output_root, copy_images=args.copy_images)
    convert_split("val", Path(args.val_root), output_root, copy_images=args.copy_images)

    print(f"Saved YOLO dataset to: {output_root}")
    print("Merged classes: truck -> big_car, bus -> big_car")


if __name__ == "__main__":
    main()
