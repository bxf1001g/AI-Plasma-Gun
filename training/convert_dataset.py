"""
Convert kakitamedia/drone_dataset JSON annotations → YOLO format

Dataset source:
  Images:      https://drive.google.com/drive/folders/11H30-Oh_Ybi_LzsRot2soHaNp2ZWlt4i
  Annotations: https://drive.google.com/file/d/1P-yM34AjsRXFDyOzGW7MbJnpuP3f3IKy

Original format:
  [{"path": "images/train/xxx.jpg", "bbox": [[x,y,w,h], ...], "label": ["hawk", ...]}]

YOLO format (per image .txt):
  <class_id> <cx> <cy> <w> <h>   (all normalized 0-1)

Classes:
  0 = hawk
  1 = crow
  2 = wild_bird
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_JSON   = "train_annotations.json"   # downloaded annotation file
VAL_JSON     = "val_annotations.json"
IMAGES_DIR   = "images"                   # root folder of downloaded images
OUTPUT_DIR   = "dataset"                  # YOLO-format output
IMAGE_W      = 3840                       # dataset image resolution
IMAGE_H      = 2160

CLASS_MAP = {
    "hawk":      0,
    "crow":      1,
    "wild bird": 2,
    "wild_bird": 2,
}

def convert(json_path: str, split: str):
    out_img_dir  = Path(OUTPUT_DIR) / "images" / split
    out_lbl_dir  = Path(OUTPUT_DIR) / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    skipped = 0
    for entry in tqdm(data, desc=f"Converting {split}"):
        img_path = Path(IMAGES_DIR) / entry["path"]
        bboxes   = entry.get("bbox", [])
        labels   = entry.get("label", [])

        if not img_path.exists():
            skipped += 1
            continue

        lines = []
        for bbox, label in zip(bboxes, labels):
            cls = CLASS_MAP.get(label.lower().strip())
            if cls is None:
                continue
            x, y, w, h = bbox
            # YOLO needs center x,y normalized
            cx = (x + w / 2) / IMAGE_W
            cy = (y + h / 2) / IMAGE_H
            nw = w / IMAGE_W
            nh = h / IMAGE_H
            # clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            skipped += 1
            continue

        # Copy image
        dest_img = out_img_dir / img_path.name
        shutil.copy2(img_path, dest_img)

        # Write label file
        lbl_file = out_lbl_dir / (img_path.stem + ".txt")
        lbl_file.write_text("\n".join(lines))

    print(f"  [{split}] Done. Skipped: {skipped}")

def write_yaml():
    yaml_content = f"""# Bird Detection Dataset — converted from kakitamedia/drone_dataset
path: {str(Path(OUTPUT_DIR).resolve())}
train: images/train
val:   images/val

nc: 3
names:
  0: hawk
  1: crow
  2: wild_bird
"""
    Path(OUTPUT_DIR, "dataset.yaml").write_text(yaml_content)
    print(f"Wrote {OUTPUT_DIR}/dataset.yaml")

if __name__ == "__main__":
    print("=== kakitamedia → YOLO converter ===")
    if Path(TRAIN_JSON).exists():
        convert(TRAIN_JSON, "train")
    else:
        print(f"WARNING: {TRAIN_JSON} not found — skipping train split")

    if Path(VAL_JSON).exists():
        convert(VAL_JSON, "val")
    else:
        print(f"WARNING: {VAL_JSON} not found — skipping val split")

    write_yaml()
    print("\nDone! Run training with:")
    print("  yolo train model=yolov8s.pt data=dataset/dataset.yaml imgsz=1280 epochs=50 batch=8")
