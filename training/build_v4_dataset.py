"""Build combined_dataset_v4: merge v3 dataset + Project 6 (DJI_0578) reviewed labels.

Steps:
  1. Pull project 6 reviewed annotations from Label Studio (only annotated tasks).
  2. Convert each annotation to YOLO labels (class 0=bird, 1=idle_bird).
  3. Hardlink/copy images + write labels into combined_dataset_v4/{train,val}.
  4. Reuse v3 train/val images and labels (already curated) in v4.
  5. Write dataset.yaml.
"""
import os
import random
import shutil
from pathlib import Path
from urllib.parse import unquote

import requests
from PIL import Image
from tqdm import tqdm

LS_URL  = "http://localhost:8080"
TOKEN   = "5a32889a9a833d4186ba64b2632bc221ee0ca83e"
PID     = 6
LS_ROOT = Path(r"D:\datasets\birds")

V3_DIR  = LS_ROOT / "combined_dataset_v3"
V4_DIR  = LS_ROOT / "combined_dataset_v4"
VAL_RATIO = 0.15
SEED = 42

CLASS_MAP = {"bird": 0, "idle_bird": 1}

H = {"Authorization": f"Token {TOKEN}"}


def fetch_annotated_tasks():
    """Get all tasks with annotations from project 6."""
    print(f"Fetching annotated tasks from project {PID}...")
    out = []
    page = 1
    while True:
        r = requests.get(
            f"{LS_URL}/api/projects/{PID}/tasks",
            headers=H,
            params={"page_size": 200, "page": page},
        )
        data = r.json()
        tasks = data if isinstance(data, list) else data.get("results", [])
        if not tasks:
            break
        out.extend(tasks)
        if len(tasks) < 200:
            break
        page += 1
    annotated = [t for t in out if t.get("annotations")]
    print(f"  total tasks: {len(out)} | with annotations: {len(annotated)}")
    return annotated


def task_image_path(task):
    url = task["data"].get("image", "")
    if "d=" not in url:
        return None
    rel = unquote(url.split("d=")[-1])
    return LS_ROOT / rel


def yolo_lines_from_annotation(ann_result, img_w, img_h):
    """Convert a Label Studio annotation 'result' list to YOLO label lines."""
    lines = []
    for r in ann_result:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        labels = v.get("rectanglelabels") or []
        if not labels:
            continue
        cls = CLASS_MAP.get(labels[0].lower())
        if cls is None:
            continue
        # LS gives x,y,width,height in percent of original image
        x = v["x"] / 100.0
        y = v["y"] / 100.0
        w = v["width"] / 100.0
        h = v["height"] / 100.0
        cx = x + w / 2
        cy = y + h / 2
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w  = min(max(w,  0.0), 1.0)
        h  = min(max(h,  0.0), 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def reuse_v3():
    print("\nReusing v3 train/val images & labels...")
    for split in ("train", "val"):
        src_img = V3_DIR / split / "images"
        src_lbl = V3_DIR / split / "labels"
        dst_img = V4_DIR / split / "images"
        dst_lbl = V4_DIR / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        imgs = list(src_img.glob("*.*"))
        for img in tqdm(imgs, desc=f"  v3 {split}"):
            link_or_copy(img, dst_img / img.name)
            lbl = src_lbl / (img.stem + ".txt")
            if lbl.exists():
                link_or_copy(lbl, dst_lbl / lbl.name)


def add_project6(tasks):
    print(f"\nAdding {len(tasks)} reviewed tasks from project 6...")
    random.seed(SEED)
    random.shuffle(tasks)
    cut = int(len(tasks) * (1 - VAL_RATIO))
    splits = [("train", tasks[:cut]), ("val", tasks[cut:])]

    added = skipped_no_img = 0
    for split, group in splits:
        dst_img = V4_DIR / split / "images"
        dst_lbl = V4_DIR / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for t in tqdm(group, desc=f"  p6 {split}"):
            img_path = task_image_path(t)
            if not img_path or not img_path.exists():
                skipped_no_img += 1
                continue
            try:
                with Image.open(img_path) as im:
                    iw, ih = im.size
            except Exception:
                skipped_no_img += 1
                continue

            # Use the latest annotation (skip cancelled)
            anns = [a for a in t["annotations"] if not a.get("was_cancelled")]
            if not anns:
                continue
            ann = anns[-1]
            lines = yolo_lines_from_annotation(ann.get("result", []), iw, ih)

            new_name = f"p6_{img_path.stem}.jpg"
            link_or_copy(img_path, dst_img / new_name)
            (dst_lbl / f"p6_{img_path.stem}.txt").write_text("\n".join(lines))
            added += 1

    print(f"  added: {added} | skipped (no image): {skipped_no_img}")


def write_yaml():
    yaml_path = V4_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {V4_DIR}\n"
        "train: train/images\n"
        "val: val/images\n"
        "nc: 2\n"
        "names: [bird, idle_bird]\n"
    )
    print(f"\nWrote {yaml_path}")


def summary():
    for split in ("train", "val"):
        n_img = len(list((V4_DIR / split / "images").glob("*.*")))
        n_lbl = len(list((V4_DIR / split / "labels").glob("*.txt")))
        print(f"  {split}: {n_img} images | {n_lbl} labels")


if __name__ == "__main__":
    V4_DIR.mkdir(parents=True, exist_ok=True)
    reuse_v3()
    tasks = fetch_annotated_tasks()
    add_project6(tasks)
    write_yaml()
    print("\nFinal v4 dataset:")
    summary()
