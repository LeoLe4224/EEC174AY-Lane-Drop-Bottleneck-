# Training and pushing this repo

To keep this repository usable on GitHub for retraining with VisDrone, push the code, config, and small reference assets. Do not push local runtime output or the raw VisDrone download.

## Push these

- `train.py`
- `configs/visdrone.yaml`
- `requirements.txt`
- `.gitignore`
- the existing helper scripts: `extract_frames.py`, `make_seed_set.py`, `auto_label.py`, `split_seed_dataset.py`
- your existing tracked sample dataset under `traffic_dataset/`
- the starting YOLO weights you want available in the repo: `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`

## Do not push these

- `datasets/visdrone/` raw images and labels
- `runs/` and `runs_highway/` training outputs
- `.venv/`, `.idea/`, `__pycache__/`
- `input_videos/`, `out_videos/`
- local build artifacts such as `a.out`

## Local VisDrone layout

Download and prepare VisDrone locally so the dataset matches:

```text
datasets/
  visdrone/
    images/
      train/
      val/
    labels/
      train/
      val/
```

`configs/visdrone.yaml` already points to that structure.

## Install

Install a CUDA-enabled PyTorch build first, then:

```bash
pip install -r requirements.txt
```

## Train

Default VisDrone training command:

```bash
python train.py
```

Example with explicit settings:

```bash
python train.py --model yolo11m.pt --data configs/visdrone.yaml --epochs 150 --imgsz 1280 --batch 8 --device 0
```

If you need CPU training for a quick check:

```bash
python train.py --device cpu --allow-cpu
```
