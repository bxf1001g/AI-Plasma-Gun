# Models

The trained YOLO weights are **not** committed to the repo (large files).

## Get the latest model
Either:

1. Download `bird_v5.pt` from the project's GitHub Releases tab and place it here as `bird_v5.pt`, **or**
2. Train it yourself with `training/train_v5.py` and copy the output:
   ```
   D:\datasets\birds\runs\bird_v5\weights\best.pt → models\bird_v5.pt
   ```

## Available versions

| Version | mAP50 | Notes |
|---|---|---|
| bird_v1 | 51.1% | Initial dataset, YOLOv8s |
| bird_v2 | 52.5% | + auto-labeled frames |
| bird_v3 | 62.9% | + curated DJI footage, YOLOv8m |
| bird_v4 | 63.3% | + 412 reviewed Project 6 frames |
| bird_v5 | 63.3%+ | + 4 Roboflow public datasets, fine-tuned from v4 |

The Jetson controller loads whatever path is set in `jetson/config.yaml`
(`model.path`).

## Convert to TensorRT for max throughput on Jetson
```bash
python3 training/convert_to_tensorrt.py --model models/bird_v5.pt --imgsz 1280
```
Update `model.path` in `config.yaml` to point to the resulting `.engine` file.
