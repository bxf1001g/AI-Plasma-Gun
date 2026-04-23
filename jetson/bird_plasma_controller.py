"""Bird-Plasma Auto-Trigger Controller (Jetson Nano Orin Super).

Pipeline:
    Arducam (USB UVC) -> YOLO bird_v5 -> motion + tracking filter
                      -> auto-fire ESP32 plasma gun via WiFi

Web UI (Flask):  http://<jetson-ip>:5000
    * Live MJPEG stream with detection overlays
    * Confidence threshold slider
    * Motion sensitivity slider
    * Cooldown editor
    * Manual ARM / DISARM, FIRE, ABORT
    * Stats: FPS, total fires, last fire time, plasma status

Run:
    python3 jetson/bird_plasma_controller.py --config jetson/config.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml
from flask import Flask, Response, jsonify, render_template_string, request
from ultralytics import YOLO

from plasma_client import PlasmaClient

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("bird_plasma")


# ─── Shared state (thread-safe via lock) ──────────────────────────────────────
@dataclass
class State:
    cfg: dict
    armed: bool = True
    total_fires: int = 0
    last_fire_at: float = 0.0
    last_fire_text: str = ""
    fps: float = 0.0
    plasma_step: str = "OFFLINE"
    plasma_firing: bool = False
    last_frame_jpg: bytes = b""
    lock: threading.Lock = field(default_factory=threading.Lock)


# ─── Tracking helpers ─────────────────────────────────────────────────────────
def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (max(0, min(w - 1, x1)), max(0, min(h - 1, y1)),
            max(0, min(w, x2)),     max(0, min(h, y2)))


def motion_ratio(mask, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return cv2.countNonZero(roi) / float(roi.size) if roi.size else 0.0


def make_motion_mask(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    return cv2.dilate(mask, None, iterations=2)


# ─── Detection / fire loop (background thread) ────────────────────────────────
class DetectionLoop(threading.Thread):
    def __init__(self, state: State, plasma: PlasmaClient):
        super().__init__(daemon=True)
        self.state = state
        self.plasma = plasma
        self.stop_event = threading.Event()
        self.tracks: dict = {}
        self.next_track_id = 1
        self.consecutive_confirmed = 0

    def stop(self):
        self.stop_event.set()

    def open_camera(self) -> cv2.VideoCapture:
        c = self.state.cfg["camera"]
        cap = cv2.VideoCapture(c["index"])  # V4L2 on Linux/Jetson
        if c.get("fourcc"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        cap.set(cv2.CAP_PROP_FPS, c["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read from camera index {c['index']}")
        log.info("Camera ready: %dx%d @ %d fps (fourcc=%s)",
                 c["width"], c["height"], c["fps"], c.get("fourcc", "default"))
        return cap

    def maybe_fire(self, confirmed_count: int):
        cfg = self.state.cfg
        with self.state.lock:
            armed = self.state.armed
            cooldown_s = cfg["trigger"]["cooldown_s"]
            since_last = time.time() - self.state.last_fire_at
        if not armed:
            return
        if confirmed_count == 0:
            self.consecutive_confirmed = 0
            return

        self.consecutive_confirmed += 1
        if self.consecutive_confirmed < cfg["trigger"]["required_consecutive_frames"]:
            return
        if self.state.last_fire_at and since_last < cooldown_s:
            return

        # FIRE!
        log.warning("🔥 BIRD CONFIRMED — firing plasma gun")
        resp = self.plasma.fire()
        with self.state.lock:
            self.state.last_fire_text = resp.text
            if resp.ok:
                self.state.total_fires += 1
                self.state.last_fire_at = time.time()
        self.consecutive_confirmed = 0

    def run(self):
        cfg = self.state.cfg
        model = YOLO(cfg["model"]["path"])
        target_cls = cfg["model"]["target_class"]
        imgsz = cfg["model"]["imgsz"]
        device = cfg["model"]["device"]

        cap = self.open_camera()
        prev_gray = None
        last_t = time.time()
        fps_smooth = 0.0
        jpeg_q = int(cfg["ui"]["jpeg_quality"])

        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                log.warning("Camera frame read failed; retrying…")
                time.sleep(0.5)
                continue

            det_cfg = cfg["detection"]  # may have been updated via UI
            conf_thr = float(det_cfg["conf_threshold"])
            min_motion = float(det_cfg["min_motion_ratio"])
            min_move_px = float(det_cfg["min_movement_px"])
            confirm_hits = int(det_cfg["confirm_moving_hits"])
            track_max_d = float(det_cfg["track_max_distance"])
            track_max_age = int(det_cfg["track_max_age"])
            trace_len = int(det_cfg["trace_length"])

            annotated = frame.copy()
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            mmask = make_motion_mask(prev_gray, gray) if prev_gray is not None else None
            prev_gray = gray

            results = model.predict(frame, conf=conf_thr, imgsz=imgsz,
                                    device=device, verbose=False)
            r = results[0]
            h, w = frame.shape[:2]

            detections = []
            for b in r.boxes:
                cls_name = model.names[int(b.cls[0])]
                if cls_name != target_cls:
                    continue
                xyxy = clip_box([int(v) for v in b.xyxy[0].tolist()], w, h)
                detections.append({
                    "box": xyxy, "conf": float(b.conf[0]),
                    "center": center_of(xyxy),
                    "motion": motion_ratio(mmask, xyxy) if mmask is not None else 0.0,
                })

            # Track association
            used = set()
            active = set()
            for d in detections:
                best_id, best_d = None, track_max_d
                for tid, tr in self.tracks.items():
                    if tid in used or tr["age"] > track_max_age:
                        continue
                    dd = distance(tr["center"], d["center"])
                    if dd < best_d:
                        best_d, best_id = dd, tid

                is_moving = d["motion"] >= min_motion
                if best_id is None:
                    self.tracks[self.next_track_id] = {
                        "center": d["center"], "box": d["box"], "conf": d["conf"],
                        "age": 0, "trace": deque([d["center"]], maxlen=trace_len),
                        "moving_hits": 1 if is_moving else 0, "confirmed": False,
                        "first_seen": time.time(),
                    }
                    used.add(self.next_track_id); active.add(self.next_track_id)
                    self.next_track_id += 1
                else:
                    tr = self.tracks[best_id]
                    disp = distance(tr["center"], d["center"])
                    is_moving = is_moving or disp >= min_move_px
                    tr["center"] = d["center"]; tr["box"] = d["box"]
                    tr["conf"] = d["conf"]; tr["age"] = 0
                    tr["trace"].append(d["center"])
                    tr["moving_hits"] = (tr["moving_hits"] + 1) if is_moving \
                        else max(0, tr["moving_hits"] - 1)
                    tr["confirmed"] = tr["moving_hits"] >= confirm_hits
                    used.add(best_id); active.add(best_id)

            # Age out stale tracks
            stale = []
            for tid, tr in self.tracks.items():
                if tid not in active:
                    tr["age"] += 1
                if tr["age"] > track_max_age:
                    stale.append(tid)
            for tid in stale:
                del self.tracks[tid]

            # Confirmed flying birds (eligible to fire)
            confirmed = [t for t in self.tracks.values()
                         if t["confirmed"] and t["age"] == 0]

            # Daylight check
            if cfg["trigger"].get("daylight_only", False):
                if gray.mean() < cfg["trigger"]["brightness_threshold"]:
                    confirmed = []

            self.maybe_fire(len(confirmed))

            # Draw overlays
            for tid, tr in self.tracks.items():
                if tr["age"] > 0:
                    continue
                x1, y1, x2, y2 = tr["box"]
                color = (0, 255, 0) if tr["confirmed"] else (0, 200, 255)
                label = f"{'BIRD' if tr['confirmed'] else 'check'} {tr['conf']:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pts = list(tr["trace"])
                for i in range(1, len(pts)):
                    cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 255), 2)

            # FPS
            now = time.time()
            dt = now - last_t; last_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

            with self.state.lock:
                self.state.fps = fps_smooth
                # Plasma status (poll every ~1s)
                if int(now) % 1 == 0:
                    s = self.plasma.status()
                    self.state.plasma_step = s.get("step", "OFFLINE")
                    self.state.plasma_firing = bool(s.get("firing", False))

            # HUD overlay
            hud = (f"FPS {fps_smooth:5.1f} | conf {conf_thr:.2f} | "
                   f"birds {len(confirmed)} | armed {self.state.armed} | "
                   f"plasma {self.state.plasma_step} | fires {self.state.total_fires}")
            cv2.putText(annotated, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)

            # Encode for MJPEG stream
            ok2, buf = cv2.imencode(".jpg", annotated,
                                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            if ok2:
                with self.state.lock:
                    self.state.last_frame_jpg = buf.tobytes()

        cap.release()
        log.info("Detection loop stopped")


# ─── Flask web UI ─────────────────────────────────────────────────────────────
INDEX_HTML = """
<!doctype html>
<html><head><title>Bird-Plasma Controller</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:10px}
  h1{margin:0 0 10px 0;color:#0f0}
  img{width:100%;max-width:1280px;border:2px solid #0f0;border-radius:8px}
  .row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
  .card{background:#222;padding:10px;border-radius:8px;flex:1;min-width:220px}
  .stat{font-size:1.6em;color:#0f0;font-weight:bold}
  .label{color:#888;font-size:.9em}
  button{font-size:1.2em;padding:14px 22px;border:0;border-radius:8px;cursor:pointer;flex:1;margin:4px}
  .arm{background:#0a0;color:#fff} .disarm{background:#a00;color:#fff}
  .fire{background:#f60;color:#fff} .test{background:#08c;color:#fff}
  input[type=range]{width:100%}
  .sliderval{color:#0f0;font-weight:bold;font-size:1.1em}
</style></head>
<body>
<h1>🐦 Bird-Plasma Controller 🔥</h1>
<img id="stream" src="/stream">

<div class="row">
  <div class="card"><div class="label">FPS</div><div id="fps" class="stat">--</div></div>
  <div class="card"><div class="label">Total fires</div><div id="fires" class="stat">--</div></div>
  <div class="card"><div class="label">Last fire (s ago)</div><div id="lastfire" class="stat">--</div></div>
  <div class="card"><div class="label">Plasma step</div><div id="step" class="stat">--</div></div>
  <div class="card"><div class="label">Armed</div><div id="armed" class="stat">--</div></div>
</div>

<div class="row">
  <button class="arm"    onclick="arm(true)">ARM</button>
  <button class="disarm" onclick="arm(false)">DISARM</button>
  <button class="fire"   onclick="manualFire()">🔥 MANUAL FIRE</button>
</div>

<div class="card">
  <div class="label">Confidence threshold: <span id="confval" class="sliderval">0.55</span></div>
  <input type="range" id="conf" min="0.30" max="0.95" step="0.01" value="0.55"
         oninput="setVal('conf')" onchange="updateCfg()">

  <div class="label">Motion sensitivity (min ratio): <span id="motval" class="sliderval">0.003</span></div>
  <input type="range" id="mot" min="0.001" max="0.05" step="0.001" value="0.003"
         oninput="setVal('mot')" onchange="updateCfg()">

  <div class="label">Required consecutive confirmed frames: <span id="rcfval" class="sliderval">4</span></div>
  <input type="range" id="rcf" min="1" max="15" step="1" value="4"
         oninput="setVal('rcf')" onchange="updateCfg()">

  <div class="label">Cooldown (seconds between auto-fires): <span id="cdval" class="sliderval">20</span></div>
  <input type="range" id="cd" min="5" max="120" step="1" value="20"
         oninput="setVal('cd')" onchange="updateCfg()">
</div>

<div class="row">
  <button class="test" onclick="test('fan','on')">PUMP ON</button>
  <button class="test" onclick="test('fan','off')">PUMP OFF</button>
  <button class="test" onclick="test('valve','on')">GAS ON</button>
  <button class="test" onclick="test('valve','off')">GAS OFF</button>
  <button class="test" onclick="test('spark','on')">SPARK ON</button>
  <button class="test" onclick="test('spark','off')">SPARK OFF</button>
  <button class="disarm" onclick="test('all','off')">⛔ ALL OFF</button>
</div>

<script>
function setVal(id){document.getElementById(id+'val').textContent=document.getElementById(id).value}
function arm(v){fetch('/api/arm?v='+v)}
function manualFire(){if(confirm('Fire plasma gun now?'))fetch('/api/fire')}
function test(c,s){fetch('/api/test?c='+c+'&s='+s)}
function updateCfg(){
  const params = new URLSearchParams({
    conf: document.getElementById('conf').value,
    mot:  document.getElementById('mot').value,
    rcf:  document.getElementById('rcf').value,
    cd:   document.getElementById('cd').value,
  });
  fetch('/api/cfg?'+params.toString());
}
async function poll(){
  try{
    const r=await fetch('/api/status'); const j=await r.json();
    document.getElementById('fps').textContent=j.fps.toFixed(1);
    document.getElementById('fires').textContent=j.total_fires;
    document.getElementById('lastfire').textContent=j.last_fire_age==null?'--':j.last_fire_age.toFixed(1);
    document.getElementById('step').textContent=j.plasma_step;
    document.getElementById('armed').textContent=j.armed?'YES':'NO';
    document.getElementById('armed').style.color=j.armed?'#0f0':'#f44';
  }catch(e){}
  setTimeout(poll,500);
}
poll();
</script>
</body></html>
"""


def make_app(state: State, plasma: PlasmaClient, loop: DetectionLoop) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/stream")
    def stream():
        target_dt = 1.0 / max(1, int(state.cfg["ui"]["stream_fps"]))

        def gen():
            while True:
                with state.lock:
                    jpg = state.last_frame_jpg
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                time.sleep(target_dt)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def api_status():
        with state.lock:
            return jsonify({
                "fps": state.fps,
                "total_fires": state.total_fires,
                "last_fire_age": (time.time() - state.last_fire_at) if state.last_fire_at else None,
                "last_fire_text": state.last_fire_text,
                "plasma_step": state.plasma_step,
                "plasma_firing": state.plasma_firing,
                "armed": state.armed,
            })

    @app.route("/api/arm")
    def api_arm():
        v = request.args.get("v", "true").lower() == "true"
        with state.lock:
            state.armed = v
        log.info("Armed = %s", v)
        return jsonify({"armed": v})

    @app.route("/api/fire")
    def api_fire():
        resp = plasma.fire()
        with state.lock:
            state.last_fire_text = resp.text
            if resp.ok:
                state.total_fires += 1
                state.last_fire_at = time.time()
        return jsonify({"ok": resp.ok, "text": resp.text})

    @app.route("/api/test")
    def api_test():
        return jsonify({"ok": plasma.test(request.args.get("c", ""),
                                          request.args.get("s", "off"))})

    @app.route("/api/cfg")
    def api_cfg():
        with state.lock:
            d = state.cfg["detection"]
            t = state.cfg["trigger"]
            if "conf" in request.args: d["conf_threshold"] = float(request.args["conf"])
            if "mot"  in request.args: d["min_motion_ratio"] = float(request.args["mot"])
            if "rcf"  in request.args: t["required_consecutive_frames"] = int(request.args["rcf"])
            if "cd"   in request.args: t["cooldown_s"] = int(request.args["cd"])
        return jsonify({"ok": True})

    return app


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    state = State(cfg=cfg)
    plasma = PlasmaClient(host=cfg["plasma"]["host"])

    # Push & lock plasma timing on the ESP32 so the operator's phone UI
    # can't override our calibrated values.
    if plasma.lock_settings(
        pre_flush_ms=cfg["plasma"]["pre_flush_ms"],
        gas_fill_ms=cfg["plasma"]["gas_fill_ms"],
        gas_pulses=cfg["plasma"]["gas_pulses"],
        gas_pause_ms=cfg["plasma"]["gas_pause_ms"],
        settle_ms=cfg["plasma"]["settle_ms"],
        spark_ms=cfg["plasma"]["spark_ms"],
        spark_retries=cfg["plasma"]["spark_retries"],
        spark_gap_ms=cfg["plasma"]["spark_gap_ms"],
        exhaust_ms=cfg["plasma"]["exhaust_ms"],
        cooldown_ms=cfg["plasma"]["cooldown_ms"],
    ):
        log.info("ESP32 timing parameters locked")
    else:
        log.warning("Could not reach ESP32 at %s — running detection only",
                    cfg["plasma"]["host"])

    loop = DetectionLoop(state, plasma)
    loop.start()

    app = make_app(state, plasma, loop)
    log.info("Web UI: http://%s:%d  (also reachable from phone on same WiFi)",
             cfg["ui"]["host"], cfg["ui"]["port"])
    try:
        app.run(host=cfg["ui"]["host"], port=cfg["ui"]["port"],
                threaded=True, debug=False)
    finally:
        loop.stop()
        loop.join(timeout=3)


if __name__ == "__main__":
    main()
