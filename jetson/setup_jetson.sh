#!/usr/bin/env bash
# Setup script for Jetson Nano Orin Super (JetPack 6.x, Ubuntu 22.04).
# Installs system deps and Python packages for the bird-plasma controller.
set -e

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    libopencv-dev python3-opencv \
    v4l-utils \
    libgl1 libglib2.0-0

echo "=== Listing connected cameras ==="
v4l2-ctl --list-devices || true

echo "=== Creating Python venv ==="
python3 -m venv ~/plasma_venv
source ~/plasma_venv/bin/activate

echo "=== Installing Python deps ==="
pip install --upgrade pip
pip install -r jetson/requirements.txt

# Use NVIDIA's PyTorch wheel for Jetson (bundled with JetPack 6 / CUDA 12.2).
# If pip auto-installs CPU torch, replace it manually:
#   pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl

echo "=== Done ==="
echo
echo "Next steps:"
echo "  1. Place the model file at: models/bird_v5.pt"
echo "  2. Connect Jetson WiFi to 'PlasmaGun-AP' (password: plasma1234)"
echo "  3. Run:  source ~/plasma_venv/bin/activate"
echo "          python3 jetson/bird_plasma_controller.py"
echo "  4. Open the web UI:  http://<jetson-ip>:5000"
