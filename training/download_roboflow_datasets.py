"""Download 3 Roboflow datasets, remap all classes → bird(0)/idle_bird(1),
merge with combined_dataset_v4 → combined_dataset_v5, write dataset.yaml.

Usage:
    python bird_ai/download_roboflow_datasets.py --api-key YOUR_KEY

Get your free API key at: https://app.roboflow.com → top-right avatar → API Key
"""
import argparse
import os
import random
import shutil
from pathlib import Path

from roboflow import Roboflow
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
DATASETS_ROOT = Path(r"D:\datasets\birds")
V4_DIR = DATASETS_ROOT / "combined_dataset_v4"
V5_DIR = DATASETS_ROOT / "combined_dataset_v5"
DOWNLOAD_DIR = DATASETS_ROOT / "roboflow_downloads"

# ── Roboflow datasets to pull ──────────────────────────────────────────────────
# (workspace, project, version)
RF_DATASETS = [
    ("birds-oiyra",        "birds-flying-fjszz",  1),  # classification-only, will be skipped
    ("myworkplace-ail2c",  "birds-far-away-dwsfd", 1),
    ("loai",               "the-number-of-birds",  1),  # no accessible version, will be skipped
    ("zheng-qing",         "birds-far-away",       1),
    ("nycuproject",        "bird-flying-z8lwp",    1),
    ("valentin-dbs-cwlzr", "flying-birds",         1),
]

# How each Roboflow class name maps to our class index.
# Any name containing "bird" (case-insensitive) → class 0 (bird).
# Anything else is dropped (non-bird objects in some datasets).
def remap_class(name: str) -> int | None:
    n = name.lower().strip()

    # Explicitly perched/idle birds → class 1
    idle_keywords = {"posada", "idle", "perch", "perched", "sitting", "static", "idle_bird"}
    if any(k in n for k in idle_keywords):
        return 1

    # Explicitly flying birds → class 0
    flying_keywords = {"volando", "flying", "flight", "fly", "rapaz", "bird"}
    if any(k in n for k in flying_keywords):
        return 0

    # Pure digit class names in bird-only datasets → bird
    if n.isdigit():
        return 0

    # Known non-bird classes to discard
    discard = {"background", "bg", "person", "human", "car", "vehicle", "drone"}
    if n in discard:
        return None

    # All remaining classes in bird-specific datasets → treat as bird
    return 0

VAL_RATIO = 0.15
SEED = 42
CLASSES = ["bird", "idle_bird"]


# ── helpers ────────────────────────────────────────────────────────────────────
def iter_yolo_pairs(split_dir: Path):
    """Yield (image_path, label_path) pairs from a YOLO split folder."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists():
        return
    for img in img_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        yield img, lbl if lbl.exists() else None


def copy_remapped(src_img: Path, src_lbl: Path | None,
                  dst_img: Path, dst_lbl: Path,
                  class_names: list[str]) -> bool:
    """Copy image + remap labels. Returns False if no valid boxes remain."""
    lines = []
    if src_lbl and src_lbl.exists():
        for line in src_lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            orig_cls = int(parts[0])
            name = class_names[orig_cls] if orig_cls < len(class_names) else ""
            new_cls = remap_class(name)
            if new_cls is None:
                continue
            lines.append(f"{new_cls} " + " ".join(parts[1:]))

    if not lines:
        return False  # skip images with no bird annotations

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)
    dst_lbl.write_text("\n".join(lines) + "\n")
    return True


def merge_split(src_dirs: list[Path], dst_split: Path,
                prefix: str, class_names_map: dict[Path, list[str]]):
    """Merge one split (train or val) from multiple source dirs into dst_split."""
    count = 0
    for src_dir in src_dirs:
        for img, lbl in tqdm(list(iter_yolo_pairs(src_dir)), desc=f"  {src_dir.parent.name}/{src_dir.name}", leave=False):
            dst_img = dst_split / "images" / f"{prefix}{img.name}"
            dst_lbl = dst_split / "labels" / f"{prefix}{img.stem}.txt"
            names = class_names_map.get(src_dir, ["bird"])
            if copy_remapped(img, lbl, dst_img, dst_lbl, names):
                count += 1
    return count


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step (use existing files in roboflow_downloads/)")
    args = parser.parse_args()

    random.seed(SEED)

    # ── 1. Download ──────────────────────────────────────────────────────────
    rf = Roboflow(api_key=args.api_key)
    download_paths: list[Path] = []

    for workspace, project_name, version in RF_DATASETS:
        out_dir = DOWNLOAD_DIR / f"{workspace}__{project_name}"
        if args.skip_download and out_dir.exists():
            print(f"[skip] {workspace}/{project_name} v{version} — already downloaded")
            download_paths.append(out_dir)
            continue
        print(f"[download] {workspace}/{project_name} v{version} …")
        try:
            project = rf.workspace(workspace).project(project_name)
            # try yolov8 first, fall back to yolov5PyTorch
            ver = project.version(version)
            for fmt in ("yolov8", "yolov5PyTorch"):
                try:
                    ver.download(fmt, location=str(out_dir))
                    break
                except Exception as e:
                    if "invalid format" in str(e).lower() or "classification" in str(e).lower():
                        continue
                    raise
            else:
                print(f"  [skip] {workspace}/{project_name} — not an object-detection dataset (no bbox annotations)")
                continue
        except Exception as e:
            print(f"  [skip] {workspace}/{project_name} — download failed: {e}")
            continue
        download_paths.append(out_dir)

    # ── 2. Build class name map for each downloaded dataset ──────────────────
    class_names_map: dict[Path, list[str]] = {}
    for dl_path in download_paths:
        yaml_files = list(dl_path.glob("*.yaml")) + list(dl_path.glob("data.yaml"))
        names = ["bird"]  # fallback
        for yf in yaml_files:
            import yaml
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            if "names" in cfg:
                raw = cfg["names"]
                names = list(raw.values()) if isinstance(raw, dict) else raw
                break
        for split in ("train", "valid", "test"):
            split_dir = dl_path / split
            if split_dir.exists():
                class_names_map[split_dir] = names
        print(f"  {dl_path.name}: classes = {names}")

    # ── 3. Collect v4 split dirs (already clean YOLO format, class 0=bird) ──
    v4_train = V4_DIR / "train"
    v4_val   = V4_DIR / "val"
    v4_class_names = ["bird", "idle_bird"]
    for sd in (v4_train, v4_val):
        class_names_map[sd] = v4_class_names

    # ── 4. Build v5 train split ──────────────────────────────────────────────
    V5_DIR.mkdir(parents=True, exist_ok=True)
    (V5_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (V5_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (V5_DIR / "val"   / "images").mkdir(parents=True, exist_ok=True)
    (V5_DIR / "val"   / "labels").mkdir(parents=True, exist_ok=True)

    print("\n── Copying v4 data ──")
    n_train = merge_split([v4_train], V5_DIR / "train", "v4_", class_names_map)
    n_val   = merge_split([v4_val],   V5_DIR / "val",   "v4_", class_names_map)
    print(f"  v4 → {n_train} train  {n_val} val")

    # ── 5. For each RF dataset: split into train/val and copy ────────────────
    for idx, dl_path in enumerate(download_paths):
        tag = f"rf{idx+1}_"
        # prefer pre-split train/valid; fall back to pooling all splits
        rf_train_dirs, rf_val_dirs = [], []
        for split in ("train",):
            sd = dl_path / split
            if sd.exists():
                rf_train_dirs.append(sd)
        for split in ("valid", "validation", "val"):
            sd = dl_path / split
            if sd.exists():
                rf_val_dirs.append(sd)
                break

        if not rf_val_dirs:
            # No pre-made val split — pool train+test and re-split
            pool = []
            for split in ("train", "test"):
                sd = dl_path / split
                if sd.exists():
                    pool.extend(list(iter_yolo_pairs(sd)))
            random.shuffle(pool)
            cut = max(1, int(len(pool) * (1 - VAL_RATIO)))
            pool_train, pool_val = pool[:cut], pool[cut:]
            # write to temp dirs
            tmp_train = DOWNLOAD_DIR / f"_tmp_{dl_path.name}_train"
            tmp_val   = DOWNLOAD_DIR / f"_tmp_{dl_path.name}_val"
            for td, pairs in ((tmp_train, pool_train), (tmp_val, pool_val)):
                (td / "images").mkdir(parents=True, exist_ok=True)
                (td / "labels").mkdir(parents=True, exist_ok=True)
                for img, lbl in pairs:
                    shutil.copy2(img, td / "images" / img.name)
                    if lbl:
                        shutil.copy2(lbl, td / "labels" / lbl.name)
                # use original class names
                class_names_map[td] = class_names_map.get(
                    dl_path / "train", ["bird"])
            rf_train_dirs = [tmp_train]
            rf_val_dirs   = [tmp_val]

        print(f"\n── RF dataset {dl_path.name} ──")
        nt = merge_split(rf_train_dirs, V5_DIR / "train", tag, class_names_map)
        nv = merge_split(rf_val_dirs,   V5_DIR / "val",   tag, class_names_map)
        print(f"  → {nt} train  {nv} val")

    # ── 6. Count totals ──────────────────────────────────────────────────────
    n_train_total = len(list((V5_DIR / "train" / "images").iterdir()))
    n_val_total   = len(list((V5_DIR / "val"   / "images").iterdir()))
    print(f"\n✅ combined_dataset_v5: {n_train_total} train + {n_val_total} val images")

    # ── 7. Write dataset.yaml ────────────────────────────────────────────────
    yaml_text = (
        f"path: {V5_DIR.as_posix()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: 2\n"
        f"names: ['bird', 'idle_bird']\n"
    )
    (V5_DIR / "dataset.yaml").write_text(yaml_text)
    print(f"dataset.yaml written → {V5_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
