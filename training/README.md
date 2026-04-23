# Training scripts

These run on a Windows/Linux machine with an NVIDIA GPU (we use an
RTX 5060 Laptop, 8 GB VRAM). Not needed on the Jetson.

## Workflow

```
1. Extract frames from drone videos:
     python extract_video_frames.py
2. Auto-label with a base model:
     python auto_label.py
3. Manually review in Label Studio
4. Build training dataset:
     python build_v4_dataset.py
5. (Optional) Pull more public data:
     python download_roboflow_datasets.py --api-key <KEY>
6. Train:
     python train_v4.py        # or train_v5.py
7. Watch training live:
     python viewresults.py --run bird_v5
```

## Model evolution

| Run | Source | Train/Val | Best mAP50 |
|---|---|---|---|
| bird_v1 | drone | 367/64 | 51.1% |
| bird_v2 | + auto-label | 552/82 | 52.5% |
| bird_v3 | + curated DJI | 579/103 | 62.9% |
| bird_v4 | + Project 6 review | 929/165 | 63.3% |
| bird_v5 | + Roboflow datasets | 3163/591 | TBD |

## Convert for Jetson
```bash
python convert_to_tensorrt.py --model best.pt --imgsz 1280
```
Generates `best.engine` for TensorRT inference on Jetson.
