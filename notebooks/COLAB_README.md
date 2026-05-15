# Colab Training: VisDrone Vehicle-Size YOLO

Use `train_visdrone_vehicles_colab.ipynb` in Google Colab.

## Steps

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose `File > Upload notebook`.
3. Upload `notebooks/train_visdrone_vehicles_colab.ipynb`.
4. Choose `Runtime > Change runtime type`.
5. Set hardware accelerator to `GPU`.
6. Run all cells from top to bottom.

The notebook downloads VisDrone, converts it into YOLO format, trains YOLO11, validates the trained model, and saves the run to Google Drive.

## Classes

```text
0 car
1 big_car
2 motorcycle
```

Mappings:

- `car`: normal sedan/SUV-sized vehicles
- `big_car`: VisDrone van, truck, and bus
- `motorcycle`: VisDrone bicycle, tricycle, awning-tricycle, and motor

## Output

The trained weight will be in Google Drive at:

```text
MyDrive/visdrone_yolo_runs/visdrone_vehicles_yolo11s/weights/best.pt
```

Download or copy that file back into the local project at:

```text
C:\Users\leo42\PycharmProjects\PythonProject\runs_highway\visdrone_vehicles_ft\weights\best.pt
```

## If Colab Runs Out Of Memory

In the settings cell, lower:

```python
BATCH = 2
```

If it still runs out of memory, lower:

```python
IMG_SIZE = 960
```
