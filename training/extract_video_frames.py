#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def collect_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise SystemExit(f"Unsupported video type: {input_path}")
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_frames(video_path: Path, output_dir: Path, target_fps: float) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if source_fps <= 0:
        source_fps = 30.0

    frame_interval = max(int(round(source_fps / target_fps)), 1)
    written = 0
    frame_index = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            frame_name = f"{video_path.stem}_{written:05d}.jpg"
            frame_path = output_dir / frame_name
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Failed to write frame: {frame_path}")
            written += 1

        frame_index += 1

    capture.release()
    return written, source_fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sampled frames from bird videos")
    parser.add_argument("--input", default=r"D:\datasets\birds", help="Input directory containing videos")
    parser.add_argument("--output", default=r"D:\datasets\birds\frames", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2.0, help="Extraction rate in frames per second")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing extracted frames for a video before extracting again",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    videos = collect_videos(input_path)
    if not videos:
        raise SystemExit(f"No videos found in {input_path}")

    total_frames = 0
    print(f"Found {len(videos)} videos")
    print(f"Extracting at {args.fps:g} fps to {output_dir}")
    print()

    for index, video_path in enumerate(videos, start=1):
        video_output_dir = output_dir / video_path.stem

        if video_output_dir.exists() and args.overwrite:
            for old_frame in video_output_dir.glob("*.jpg"):
                old_frame.unlink()

        if video_output_dir.exists() and any(video_output_dir.glob("*.jpg")) and not args.overwrite:
            existing = sum(1 for _ in video_output_dir.glob("*.jpg"))
            print(f"[{index:02d}/{len(videos)}] {video_path.name}: skipped ({existing} existing frames)")
            total_frames += existing
            continue

        written, source_fps = extract_frames(video_path, video_output_dir, args.fps)
        total_frames += written
        print(
            f"[{index:02d}/{len(videos)}] {video_path.name}: "
            f"{written} frames extracted from {source_fps:.2f} fps source"
        )

    print()
    print(f"Done. Total extracted frames: {total_frames}")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
