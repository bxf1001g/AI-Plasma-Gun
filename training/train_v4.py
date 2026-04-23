"""Train bird_v4 from bird_v3 best.pt + combined_dataset_v4."""
from pathlib import Path

import torch
from ultralytics import YOLO

DATA   = Path(r"D:\datasets\birds\combined_dataset_v4\dataset.yaml")
WEIGHTS_INIT = Path(r"D:\datasets\birds\runs\bird_v3\weights\best.pt")
PROJECT = Path(r"D:\datasets\birds\runs")
NAME    = "bird_v4"

if __name__ == "__main__":
    assert DATA.exists(),    f"missing {DATA}"
    assert WEIGHTS_INIT.exists(), f"missing {WEIGHTS_INIT}"

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU  : {torch.cuda.get_device_name(0)}  VRAM {vram:.1f}GB")
    batch = 4 if vram < 10 else 8

    model = YOLO(str(WEIGHTS_INIT))
    model.train(
        data       = str(DATA),
        epochs     = 100,
        imgsz      = 1280,
        batch      = batch,
        device     = 0,
        workers    = 4,
        project    = str(PROJECT),
        name       = NAME,
        exist_ok   = True,
        amp        = True,
        patience   = 25,
        optimizer  = "AdamW",
        lr0        = 0.0008,   # lower than v3 since fine-tuning
        lrf        = 0.01,
        warmup_epochs = 2,
        hsv_h      = 0.015,
        hsv_s      = 0.5,
        hsv_v      = 0.4,
        flipud     = 0.2,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.1,
        copy_paste = 0.1,
        close_mosaic = 10,
    )
    best = PROJECT / NAME / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
