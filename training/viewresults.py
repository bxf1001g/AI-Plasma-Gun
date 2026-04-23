import argparse
import csv
import time
from pathlib import Path


DEFAULT_RUNS_DIR = Path(r"D:\datasets\birds\runs")
DEFAULT_RUN_NAME = "bird_v4"
DEFAULT_COMPARE_RUNS = ["bird_v1", "bird_v2", "bird_v3", "bird_v4"]
DEFAULT_PATIENCE = 25


def read_rows(csv_path: Path):
    if not csv_path.exists():
        return []

    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("epoch"):
                rows.append(row)
    return rows


def best_map50(csv_path: Path):
    rows = read_rows(csv_path)
    if not rows:
        return None
    return max(float(r["metrics/mAP50(B)"]) for r in rows) * 100.0


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:5.1f}%"


def fmt_secs(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(description="Watch YOLO training progress from results.csv")
    parser.add_argument("--run", default=DEFAULT_RUN_NAME, help="Run name inside D:\\datasets\\birds\\runs")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=DEFAULT_COMPARE_RUNS,
        help="Previous runs to compare against (space-separated)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Target epoch count")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early-stop patience")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    args = parser.parse_args()

    run_dir = DEFAULT_RUNS_DIR / args.run
    csv_path = run_dir / "results.csv"
    baselines = {}
    for name in args.compare:
        m = best_map50(DEFAULT_RUNS_DIR / name / "results.csv")
        if m is not None:
            baselines[name] = m

    print(f"Watching training: {run_dir}")
    print(f"CSV file        : {csv_path}")
    if baselines:
        for name, m in baselines.items():
            print(f"Baseline {name:8s}: best mAP50 = {m:.1f}%")
    else:
        print("Baselines      : (none found)")
    print()

    seen_rows = 0
    while True:
        rows = read_rows(csv_path)
        if not rows:
            print("Waiting for results.csv...", flush=True)
            time.sleep(args.interval)
            continue

        if len(rows) == seen_rows:
            time.sleep(args.interval)
            continue

        current = rows[-1]
        epoch = int(float(current["epoch"]))
        elapsed = float(current["time"])
        best_row = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
        best_epoch = int(float(best_row["epoch"]))
        best_map = float(best_row["metrics/mAP50(B)"]) * 100.0
        epochs_since_best = epoch - best_epoch
        avg_epoch_time = elapsed / max(epoch, 1)
        eta = avg_epoch_time * max(args.epochs - epoch, 0)

        print("=" * 88)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"P {fmt_pct(float(current['metrics/precision(B)']))} | "
            f"R {fmt_pct(float(current['metrics/recall(B)']))} | "
            f"mAP50 {fmt_pct(float(current['metrics/mAP50(B)']))} | "
            f"mAP50-95 {fmt_pct(float(current['metrics/mAP50-95(B)']))}"
        )
        print(
            f"Train loss: box {float(current['train/box_loss']):.3f} | "
            f"cls {float(current['train/cls_loss']):.3f} | "
            f"dfl {float(current['train/dfl_loss']):.3f}"
        )
        print(
            f"Val loss  : box {float(current['val/box_loss']):.3f} | "
            f"cls {float(current['val/cls_loss']):.3f} | "
            f"dfl {float(current['val/dfl_loss']):.3f}"
        )
        patience_marker = "  <-- EARLY STOP" if epochs_since_best >= args.patience else ""
        print(
            f"Best so far: epoch {best_epoch:3d} | mAP50 {best_map:5.1f}% | "
            f"elapsed {fmt_secs(elapsed)} | ETA {fmt_secs(eta)}"
        )
        print(
            f"Patience   : {epochs_since_best:2d}/{args.patience} epochs since best{patience_marker}"
        )
        if baselines:
            parts = []
            for name, m in baselines.items():
                delta = best_map - m
                sign = "+" if delta >= 0 else ""
                parts.append(f"{name} {sign}{delta:.1f}%")
            print("Vs baseline: " + " | ".join(parts))
        print()

        seen_rows = len(rows)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
