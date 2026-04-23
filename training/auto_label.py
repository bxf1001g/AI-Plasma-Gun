"""
Auto-label bird images using pre-trained YOLOv8 and push predictions to Label Studio.
Runs YOLO locally on images, then imports bounding box predictions via the LS API.
"""
import argparse
import json
import os
import requests
from pathlib import Path
from ultralytics import YOLO


# COCO class 14 = "bird"
BIRD_CLASS_ID = 14


def run_yolo_on_images(image_dir: str, model_name: str = "yolov8m.pt", conf: float = 0.15, imgsz: int = 1280):
    """Run YOLO detection on all images and return predictions."""
    model = YOLO(model_name)
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob("*.jpg"))
    print(f"Running {model_name} on {len(images)} images (conf={conf}, imgsz={imgsz})...")

    predictions = {}
    bird_count = 0

    for i, img_path in enumerate(images):
        results = model.predict(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        result = results[0]

        boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == BIRD_CLASS_ID:
                # YOLO returns xyxy in pixels, convert to Label Studio format (percentages)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                boxes.append({
                    "x": (x1 / img_w) * 100,
                    "y": (y1 / img_h) * 100,
                    "width": ((x2 - x1) / img_w) * 100,
                    "height": ((y2 - y1) / img_h) * 100,
                    "score": float(box.conf[0]),
                    "label": "bird"
                })

        predictions[img_path.name] = boxes
        bird_count += len(boxes)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(images)} images, {bird_count} birds found so far")

    print(f"Done! Found {bird_count} birds across {len(images)} images")
    return predictions


def push_to_label_studio(predictions: dict, project_id: int, ls_url: str, api_token: str):
    """Push YOLO predictions to Label Studio as pre-annotations."""
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }

    # Get all tasks in the project
    print("Fetching tasks from Label Studio...")
    all_tasks = []
    page = 1
    while True:
        resp = requests.get(
            f"{ls_url}/api/projects/{project_id}/tasks",
            headers=headers,
            params={"page_size": 100, "page": page}
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            all_tasks.extend(data)
            if len(data) < 100:
                break
        elif isinstance(data, dict) and "tasks" in data:
            all_tasks.extend(data["tasks"])
            if len(data["tasks"]) < 100:
                break
        else:
            all_tasks.append(data)
            break
        page += 1

    print(f"Found {len(all_tasks)} tasks")

    # Build filename -> task_id mapping
    task_map = {}
    for task in all_tasks:
        image_url = task.get("data", {}).get("image", "")
        # Extract filename from URL like /data/local-files/?d=frames_trimmed%5Cfilename.jpg
        filename = image_url.split("%5C")[-1].split("/")[-1]
        if "%2F" in filename:
            filename = filename.split("%2F")[-1]
        task_map[filename] = task["id"]

    # Push predictions
    pushed = 0
    skipped = 0
    for filename, boxes in predictions.items():
        task_id = task_map.get(filename)
        if not task_id:
            skipped += 1
            continue

        if not boxes:
            continue

        # Build Label Studio prediction format
        results = []
        for box in boxes:
            results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": box["x"],
                    "y": box["y"],
                    "width": box["width"],
                    "height": box["height"],
                    "rectanglelabels": [box["label"]]
                },
                "score": box["score"]
            })

        pred_body = {
            "task": task_id,
            "result": results,
            "score": sum(b["score"] for b in boxes) / len(boxes),
            "model_version": "yolov8m-coco"
        }

        resp = requests.post(
            f"{ls_url}/api/predictions",
            headers=headers,
            json=pred_body
        )
        if resp.status_code == 201:
            pushed += 1
        else:
            print(f"  Error for task {task_id}: {resp.status_code} {resp.text[:100]}")

    print(f"Pushed {pushed} predictions, skipped {skipped} (no matching task)")


def main():
    parser = argparse.ArgumentParser(description="Auto-label birds with YOLOv8")
    parser.add_argument("--image-dir", default=r"D:\datasets\birds\frames_trimmed", help="Image directory")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model name")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--project", type=int, default=2, help="Label Studio project ID")
    parser.add_argument("--ls-url", default="http://localhost:8081", help="Label Studio URL")
    parser.add_argument("--api-token", default="5a32889a9a833d4186ba64b2632bc221ee0ca83e", help="LS API token")
    parser.add_argument("--save-json", default=None, help="Save predictions to JSON file")
    args = parser.parse_args()

    # Run YOLO
    predictions = run_yolo_on_images(args.image_dir, args.model, args.conf, args.imgsz)

    # Optionally save to JSON
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to {args.save_json}")

    # Push to Label Studio
    push_to_label_studio(predictions, args.project, args.ls_url, args.api_token)


if __name__ == "__main__":
    main()
