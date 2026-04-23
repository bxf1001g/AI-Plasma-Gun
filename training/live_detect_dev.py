"""Live bird detection using laptop camera with bird_v3 model.

Filters to:
- confidence >= 0.60
- class "bird" only
- moving detections only
- trace lines for tracked flying birds
"""

from collections import deque
import math

import cv2
from ultralytics import YOLO


MODEL_PATH = r"D:\datasets\birds\runs\bird_v5\weights\best.pt"
CONF_THRESHOLD = 0.40
IMG_SIZE = 1280
TARGET_CLASS = "bird"

MIN_MOTION_RATIO = 0.002
MIN_MOVEMENT_PX = 5
TRACK_MAX_DISTANCE = 20
TRACK_MAX_AGE = 12
TRACE_LENGTH = 10
CONFIRM_MOVING_HITS = 2
CAMERA_CANDIDATES = [
    (0, cv2.CAP_DSHOW, "camera 0 via DirectShow"),
    (1, cv2.CAP_DSHOW, "camera 1 via DirectShow"),
    (0, cv2.CAP_ANY, "camera 0 via default backend"),
    (1, cv2.CAP_ANY, "camera 1 via default backend"),
]


def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def motion_ratio(mask, box):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return cv2.countNonZero(roi) / float(roi.size)


def make_motion_mask(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def match_track_id(tracks, detection_center, used_track_ids):
    best_track_id = None
    best_distance = TRACK_MAX_DISTANCE

    for track_id, track in tracks.items():
        if track_id in used_track_ids or track["age"] > TRACK_MAX_AGE:
            continue

        d = distance(track["center"], detection_center)
        if d < best_distance:
            best_distance = d
            best_track_id = track_id

    return best_track_id


def open_camera():
    for index, backend, label in CAMERA_CANDIDATES:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, label

        cap.release()

    raise RuntimeError(
        "Could not open the webcam. Close other apps using the camera "
        "(Camera, WhatsApp, Zoom, browser), then try again."
    )


model = YOLO(MODEL_PATH)
cap, camera_label = open_camera()

tracks = {}
next_track_id = 1
prev_gray = None

print("Bird Detection Live — Press 'q' to quit")
print(f"Model: {MODEL_PATH}")
print(f"Confidence threshold: {CONF_THRESHOLD:.2f}")
print("Showing only moving flying birds with traces")
print(f"Using {camera_label}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Camera frame read failed. Reopen the camera and run again.")
        break

    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    motion_mask = None
    if prev_gray is not None:
        motion_mask = make_motion_mask(prev_gray, gray)
    prev_gray = gray

    results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    result = results[0]

    frame_h, frame_w = frame.shape[:2]
    raw_detections = []
    detections = []

    for box in result.boxes:
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]
        if cls_name != TARGET_CLASS:
            continue

        xyxy = [int(v) for v in box.xyxy[0].tolist()]
        xyxy = clip_box(xyxy, frame_w, frame_h)
        raw_detections.append(
            {
                "box": xyxy,
                "conf": conf,
                "center": center_of(xyxy),
            }
        )

    if motion_mask is not None:
        for detection in raw_detections:
            ratio = motion_ratio(motion_mask, detection["box"])
            detections.append(
                {
                    "box": detection["box"],
                    "conf": detection["conf"],
                    "motion_ratio": ratio,
                    "center": detection["center"],
                }
            )

    used_track_ids = set()
    active_track_ids = set()

    for detection in detections:
        track_id = match_track_id(tracks, detection["center"], used_track_ids)
        is_moving = detection["motion_ratio"] >= MIN_MOTION_RATIO

        if track_id is None:
            track_id = next_track_id
            next_track_id += 1
            tracks[track_id] = {
                "center": detection["center"],
                "box": detection["box"],
                "conf": detection["conf"],
                "age": 0,
                "trace": deque([detection["center"]], maxlen=TRACE_LENGTH),
                "moving_hits": 1 if is_moving else 0,
                "confirmed": False,
            }
        else:
            track = tracks[track_id]
            displacement = distance(track["center"], detection["center"])
            is_moving = is_moving or displacement >= MIN_MOVEMENT_PX

            track["center"] = detection["center"]
            track["box"] = detection["box"]
            track["conf"] = detection["conf"]
            track["age"] = 0
            track["trace"].append(detection["center"])

            if is_moving:
                track["moving_hits"] += 1
            else:
                track["moving_hits"] = max(0, track["moving_hits"] - 1)

            track["confirmed"] = track["moving_hits"] >= CONFIRM_MOVING_HITS

        used_track_ids.add(track_id)
        active_track_ids.add(track_id)

    stale_track_ids = []
    for track_id, track in tracks.items():
        if track_id not in active_track_ids:
            track["age"] += 1
        if track["age"] > TRACK_MAX_AGE:
            stale_track_ids.append(track_id)

    for track_id in stale_track_ids:
        del tracks[track_id]

    flying_count = 0
    raw_count = len(raw_detections)
    moving_candidate_count = sum(1 for d in detections if d["motion_ratio"] >= MIN_MOTION_RATIO)

    for track_id, track in tracks.items():
        if track["age"] > 0 or not track["confirmed"]:
            continue

        flying_count += 1
        x1, y1, x2, y2 = track["box"]
        label = f"Flying bird {track['conf']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        points = list(track["trace"])
        for i in range(1, len(points)):
            thickness = max(1, 4 - (len(points) - i) // 6)
            cv2.line(annotated, points[i - 1], points[i], (0, 255, 255), thickness)

        cv2.circle(annotated, track["center"], 4, (0, 255, 255), -1)

    header = (
        f"Flying birds: {flying_count}  Raw birds: {raw_count}  "
        f"Moving: {moving_candidate_count}  Conf >= {CONF_THRESHOLD:.2f}"
    )
    cv2.putText(annotated, header, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(
        annotated,
        "Green boxes are moving birds; raw detections are counted above",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Bird Detection Live", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
