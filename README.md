# 🐦🔥 Plasma AI — Bird-deterrent system for shrimp ponds

End-to-end autonomous bird-deterrent for shrimp farms:

```
 ┌─────────────────┐    USB UVC     ┌────────────────────────┐    WiFi    ┌────────────────┐
 │ Arducam IMX662  │  ───────────►  │  Jetson Nano Orin S.   │ ─────────► │ ESP32-S3       │
 │ (low-light cam) │                │  YOLOv8 bird_v5        │   HTTP     │ Plasma Gun     │
 └─────────────────┘                │  motion + tracking     │            │ (LPG + spark)  │
                                    │  Flask web UI :5000    │            │ AP 192.168.4.1 │
                                    └────────────────────────┘            └────────────────┘
```

The Jetson watches the pond, classifies flying birds, and only fires the LPG
combustion "plasma gun" when a bird is **(a)** confidently detected, **(b)**
verified to be moving, and **(c)** confirmed across multiple consecutive
frames. The loud blast scares birds away from the shrimp.

## Repo layout
```
plasam_ai/
├── jetson/               # Runs on the Jetson Nano Orin Super
│   ├── bird_plasma_controller.py   # main process (detection + auto-fire + web UI)
│   ├── plasma_client.py            # HTTP client for the ESP32
│   ├── config.yaml                 # all tunable parameters
│   ├── requirements.txt
│   ├── setup_jetson.sh
│   └── README.md
├── esp32/                # ESP32-S3 firmware (Arduino)
│   ├── plasma_gun.ino
│   └── README.md
├── training/             # Run on your dev PC with NVIDIA GPU
│   ├── train_v4.py / train_v5.py   # fine-tune YOLOv8
│   ├── build_v4_dataset.py
│   ├── download_roboflow_datasets.py
│   ├── viewresults.py              # live training monitor
│   ├── convert_to_tensorrt.py      # optimize for Jetson
│   └── README.md
├── models/               # Trained weights (not committed; see README)
│   └── README.md
└── docs/                 # Wiring diagrams, photos, etc.
```

## Quick start

### On the dev PC (one-time training)
```bash
cd training
pip install -r requirements.txt
python train_v5.py
```

### On the Jetson Nano Orin Super
```bash
git clone https://github.com/bxf1001g/AI-Plasma-Gun.git
cd AI-Plasma-Gun
./jetson/setup_jetson.sh

# Download trained weights to models/bird_v5.pt
# Connect Jetson WiFi to 'PlasmaGun-AP' (password: plasma1234)

source ~/plasma_venv/bin/activate
python3 jetson/bird_plasma_controller.py
```

Open the web UI on any phone connected to the same WiFi:
```
http://<jetson-ip>:5000
```

### On the ESP32-S3 (one-time flash)
```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3 esp32/plasma_gun.ino
arduino-cli upload  --fqbn esp32:esp32:esp32s3 -p /dev/ttyUSB0 esp32/plasma_gun.ino
```

## Hardware

| Component | Notes |
|---|---|
| Camera | Arducam B0576 2.4 MP IMX662 ultra-low-light USB UVC |
| Inference | NVIDIA Jetson Nano Orin Super (JetPack 6.x) |
| Plasma controller | ESP32-S3 |
| Spark | DC 5 V → 400 kV high-voltage coil (GPIO 21) |
| Gas valve | 12 V solenoid for LPG (GPIO 6, via relay) |
| Vacuum pump | 12 V 15 L/min (GPIO 3, via relay) |
| Combustion chamber | 140 mm PVC pipe |

## Safety
This system controls a **gas combustion device**. Always:

* DISARM the system from the web UI before any maintenance
* Test components individually with the manual buttons first
* Use a long EXHAUST time (≥ 15 s) so unburnt LPG cannot accumulate
* Mount the chamber pointing **away** from people, animals, and structures
* Keep an emergency power cut switch within reach during testing

The Jetson controller pushes locked timing parameters to the ESP32 at every
startup so an operator's phone connected to the ESP32 AP cannot bypass them.

## License
MIT (see LICENSE).
