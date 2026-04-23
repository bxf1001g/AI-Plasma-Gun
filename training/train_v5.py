"""Train bird_v5 from bird_v4 best.pt + combined_dataset_v5 (v4 + 4 Roboflow datasets).

Run AFTER:
  1. bird_v4 training completes
  2. python bird_ai/download_roboflow_datasets.py --api-key YOUR_KEY
"""
from pathlib import Path

import torch
from ultralytics import YOLO

DATA         = Path(r"D:\datasets\birds\combined_dataset_v5\dataset.yaml")
WEIGHTS_INIT = Path(r"D:\datasets\birds\runs\bird_v4\weights\best.pt")
PROJECT      = Path(r"D:\datasets\birds\runs")
NAME         = "bird_v5"

if __name__ == "__main__":
    assert DATA.exists(),         f"missing {DATA}\n  → run bird_ai/download_roboflow_datasets.py first"
    assert WEIGHTS_INIT.exists(), f"missing {WEIGHTS_INIT}\n  → wait for bird_v4 to finish"

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU  : {torch.cuda.get_device_name(0)}  VRAM {vram:.1f}GB")
    batch = 4 if vram < 10 else 8

    model = YOLO(str(WEIGHTS_INIT))
    model.train(
        data          = str(DATA),
        epochs        = 100,
        imgsz         = 1280,
        batch         = batch,
        device        = 0,
        workers       = 4,
        project       = str(PROJECT),
        name          = NAME,
        exist_ok      = True,
        amp           = True,
        patience      = 20,
        optimizer     = "AdamW",
        lr0           = 0.0005,   # even lower — fine-tuning from v4
        lrf           = 0.01,
        warmup_epochs = 2,
        hsv_h         = 0.015,
        hsv_s         = 0.5,
        hsv_v         = 0.4,
        flipud        = 0.2,
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.15,
        copy_paste    = 0.1,
        close_mosaic  = 10,
    )
    best = PROJECT / NAME / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
