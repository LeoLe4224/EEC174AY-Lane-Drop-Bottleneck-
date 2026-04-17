import sys
import torch
from ultralytics import YOLO

# =========================================================
# SETTINGS
# =========================================================

MODEL_START = "yolo11s.pt"
DATA_YAML = "traffic_dataset/data.yaml"
EPOCHS = 100
IMG_SIZE = 1280
BATCH = 4
PROJECT_NAME = "."
RUN_NAME = "runs_highway/car_detector_ft"
DEVICE = 0


def main():
    if not torch.cuda.is_available():
        print("CUDA GPU not available in this Python environment.")
        print("Training is blocked because CPU fallback is disabled.")
        print("Install a CUDA-enabled PyTorch build and run this script from that interpreter.")
        sys.exit(1)

    model = YOLO(MODEL_START)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        pretrained=True,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
