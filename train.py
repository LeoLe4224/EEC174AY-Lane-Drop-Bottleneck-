import argparse
import sys

import torch
from ultralytics import YOLO


DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_DATA = "configs/visdrone_vehicles.yaml"
DEFAULT_EPOCHS = 100
DEFAULT_IMG_SIZE = 1280
DEFAULT_BATCH = 4
DEFAULT_PROJECT = "."
DEFAULT_RUN_NAME = "runs_highway/visdrone_vehicles_ft"
DEFAULT_DEVICE = "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model on a local dataset such as VisDrone."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Starting weights file.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epoch count.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Training image size.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Ultralytics project output path.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Ultralytics run name.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help='CUDA device like "0". Use "cpu" to allow CPU training.',
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training on CPU when CUDA is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wants_cpu = str(args.device).lower() == "cpu"

    if not wants_cpu and not torch.cuda.is_available() and not args.allow_cpu:
        print("CUDA GPU not available in this Python environment.")
        print("Training is blocked because CPU fallback is disabled by default.")
        print('Install a CUDA-enabled PyTorch build, pass --device cpu, or add --allow-cpu.')
        sys.exit(1)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        pretrained=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
