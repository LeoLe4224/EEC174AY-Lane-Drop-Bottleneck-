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
- `datasets/visdrone_raw/` raw VisDrone downloads
- `runs/` and `runs_highway/` training outputs
- `.venv/`, `.idea/`, `__pycache__/`
- `input_videos/`, `out_videos/`
- local build artifacts such as `a.out`

## Local VisDrone download

Download the official VisDrone-DET train and val archives locally, then extract them to:

```text
datasets/
  visdrone_raw/
    VisDrone2019-DET-train/
      images/
      annotations/
    VisDrone2019-DET-val/
      images/
      annotations/
```

## Label scheme

The generated YOLO dataset uses this class list:

- `pedestrian`
- `people`
- `bicycle`
- `car`
- `van`
- `big_car`
- `tricycle`
- `awning-tricycle`
- `motor`

Raw VisDrone classes `truck` and `bus` are merged into `big_car` during conversion.

## Generated VisDrone layout

Build the training dataset with:

```bash
python prepare_visdrone.py --clear-output
```

This creates:

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
