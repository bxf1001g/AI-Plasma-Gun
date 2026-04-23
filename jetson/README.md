# Jetson — Bird-Plasma Controller

Runs on **Jetson Nano Orin Super** (JetPack 6.x). Captures video from the
**Arducam B0576 (IMX662 USB UVC)**, runs YOLO `bird_v5` for bird detection,
applies motion + tracking filters to ignore static objects, and triggers the
ESP32 plasma gun over WiFi when a flying bird is confirmed.

## Files
| File | Purpose |
|---|---|
| `bird_plasma_controller.py` | Main process — capture → detect → track → fire |
| `plasma_client.py` | HTTP client for the ESP32 firmware |
| `config.yaml` | All tunable parameters (loaded at startup) |
| `requirements.txt` | Python deps |
| `setup_jetson.sh` | One-shot install script |

## Install
```bash
git clone https://github.com/<you>/plasam_ai.git
cd plasam_ai
chmod +x jetson/setup_jetson.sh
./jetson/setup_jetson.sh
```

## Get the model
Copy `bird_v5.pt` (or download from the GitHub Release) into `models/`:
```bash
mkdir -p models
scp <your-pc>:D:/datasets/birds/runs/bird_v5/weights/best.pt models/bird_v5.pt
```

For maximum throughput, convert to TensorRT first (optional — see
`training/convert_to_tensorrt.py`).

## WiFi: Jetson ↔ ESP32
The ESP32 hosts an open SoftAP `PlasmaGun-AP` (password `plasma1234`) at
`192.168.4.1`. Connect the Jetson's WiFi to that network. The Jetson then has
**no internet**; if you need internet for both, switch the ESP32 firmware to
station mode and put both devices on your local LAN, then update
`plasma.host` in `config.yaml`.

## Run
```bash
source ~/plasma_venv/bin/activate
python3 jetson/bird_plasma_controller.py
```

Open the web UI from any phone/tablet on the same WiFi:
```
http://<jetson-ip>:5000
```

## Web UI features
* **Live MJPEG stream** with bounding boxes (orange = candidate, green = confirmed)
* **ARM / DISARM** master switch
* **Manual fire** button (with confirmation)
* **Confidence threshold slider** (0.30–0.95)
* **Motion sensitivity slider**
* **Required consecutive frames** before auto-fire (1–15)
* **Cooldown editor** (5–120 s)
* **Manual component test** buttons (PUMP, GAS, SPARK, ALL OFF)
* Live stats: FPS, total fires, last fire age, plasma step

## How "is the bird actually moving?" works
For every detection that survives the YOLO confidence filter:
1. **Motion mask** — frame-difference between consecutive frames; we count
   how many pixels inside the bbox are "moving" (`min_motion_ratio` in config)
2. **Centroid displacement** — track centers across frames; if the bird's
   centroid moves at least `min_movement_px` pixels, count as moving
3. **Track confirmation** — a detection must register as moving for
   `confirm_moving_hits` frames in a row before it's "confirmed"
4. **Trigger guard** — a confirmed bird must remain for
   `required_consecutive_frames` frames before the fire signal is sent

This kills static objects (clouds, fence posts, leaves) almost entirely. Tune
both sliders live from the UI while watching the stream.

## Auto-start on boot (systemd)
```bash
sudo tee /etc/systemd/system/bird-plasma.service >/dev/null <<EOF
[Unit]
Description=Bird-Plasma Controller
After=network.target

[Service]
User=$USER
WorkingDirectory=/home/$USER/plasam_ai
ExecStart=/home/$USER/plasma_venv/bin/python3 jetson/bird_plasma_controller.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now bird-plasma
journalctl -u bird-plasma -f
```

## Safety
* DISARM the controller before any maintenance.
* Always test with the **PUMP/GAS/SPARK** buttons individually first.
* Keep `cooldown_s` ≥ `cooldown_ms / 1000` from `plasma.cooldown_ms` so the
  controller can never out-pace the ESP32's own cooldown.
* The locked plasma timings are pushed to the ESP32 at every Jetson startup
  so a phone connected to the ESP32 AP can't override them.
