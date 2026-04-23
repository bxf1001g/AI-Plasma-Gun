# ESP32 — Plasma Gun Firmware

Hardware: **ESP32-S3** with three active-LOW relays.

| GPIO | Component |
|---|---|
| 6 | LPG solenoid valve (12 V) |
| 21 | DC 5 V → 400 kV spark coil |
| 3 | Vacuum pump 15 L/min (12 V) |

## Sequence (v3.3 — non-blocking)
```
PURGE (15 s, pump ON)
  → GAS FILL (225 ms × 1, pump still ON)
  → SETTLE (0 ms, pump still ON)
  → SPARK (500 ms × 3 retries, pump still ON)
  → EXHAUST (15 s, pump still ON)
  → PUMP OFF, READY
```

The whole sequence is now driven by a state machine in `loop()` so the WiFi
stack and HTTP server stay responsive throughout — phone apps no longer
disconnect during the EXHAUST phase.

## Build & flash
Install the ESP32 board package in arduino-cli, then:
```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3 esp32/plasma_gun.ino
arduino-cli upload  --fqbn esp32:esp32:esp32s3 -p COM3 esp32/plasma_gun.ino
```
Replace `COM3` with `/dev/ttyUSB0` (Linux) or `/dev/cu.SLAB_USBtoUART` (macOS).

## WiFi
SoftAP `PlasmaGun-AP` / password `plasma1234`. UI: <http://192.168.4.1>.

## HTTP API (used by Jetson)
| Endpoint | Notes |
|---|---|
| `GET /fire` | Start sequence. Returns `Sequence started!` / `Busy: <step>` / `Cooling down...` |
| `GET /status` | `{"firing": bool, "step": "PURGE"|"GAS FILL"|"SPARK"|"EXHAUST"|"READY"}` |
| `GET /settings?pf=...&gf=...&...` | Update timing (Jetson locks values at startup) |
| `GET /test?c=valve|spark|fan|all&s=on|off` | Manual component test |
