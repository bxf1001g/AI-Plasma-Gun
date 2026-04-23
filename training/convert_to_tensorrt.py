#!/usr/bin/env python3
"""
On-Jetson TensorRT Conversion
==============================
Run this ON the Jetson Nano Orin AFTER copying best.onnx from Kaggle.

Requirements (already on JetPack 6):
    - TensorRT 8.x
    - ONNX Runtime
    - Ultralytics

Usage:
    python convert_to_tensorrt.py --onnx best.onnx --output best.engine
"""

import argparse
import subprocess
from pathlib import Path


def convert(onnx_path: str, engine_path: str, fp16: bool = True):
    onnx_path   = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    precision = "--fp16" if fp16 else ""
    cmd = (
        f"trtexec "
        f"--onnx={onnx_path} "
        f"--saveEngine={engine_path} "
        f"{precision} "
        f"--minShapes=images:1x3x1280x1280 "
        f"--optShapes=images:1x3x1280x1280 "
        f"--maxShapes=images:1x3x1280x1280 "
        f"--workspace=4096"
    )

    print(f"Running: {cmd}\n")
    print("⏳ This takes ~15 minutes on Jetson Orin...")
    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print(f"\n✅ TensorRT engine saved: {engine_path}")
        print(f"   Use this .engine file in your bird detection inference script.")
    else:
        print(f"\n❌ Conversion failed (code {result.returncode})")
        print("   Make sure TensorRT is installed: sudo apt install tensorrt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",   default="best.onnx",   help="Input ONNX file")
    parser.add_argument("--output", default="best.engine", help="Output TensorRT engine")
    parser.add_argument("--fp32",   action="store_true",   help="Use FP32 (default FP16)")
    args = parser.parse_args()

    convert(args.onnx, args.output, fp16=not args.fp32)
