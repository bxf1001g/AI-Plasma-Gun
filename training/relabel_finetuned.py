"""Re-run auto-labeling on unlabeled images using the fine-tuned bird model."""
import requests
import os
from ultralytics import YOLO

token = '5a32889a9a833d4186ba64b2632bc221ee0ca83e'
ls_url = 'http://localhost:8080'
headers_json = {'Authorization': f'Token {token}', 'Content-Type': 'application/json'}
headers_auth = {'Authorization': f'Token {token}'}
project_id = 2
img_dir = r'D:\datasets\birds\frames_trimmed'
model_path = r'D:\datasets\birds\runs\bird_v2\weights\best.pt'

# Get all tasks
print('Fetching tasks...')
all_tasks = []
page = 1
while True:
    resp = requests.get(
        f'{ls_url}/api/projects/{project_id}/tasks',
        headers=headers_auth,
        params={'page_size': 100, 'page': page, 'fields': 'all'}
    )
    data = resp.json()
    tasks = data if isinstance(data, list) else [data]
    all_tasks.extend(tasks)
    if len(tasks) < 100:
        break
    page += 1

unlabeled = [t for t in all_tasks if t.get('total_annotations', 0) == 0]
labeled = [t for t in all_tasks if t.get('total_annotations', 0) > 0]
print(f'Total: {len(all_tasks)}, Labeled: {len(labeled)}, Unlabeled: {len(unlabeled)}')

# Delete old predictions on unlabeled tasks
deleted = 0
for idx, task in enumerate(unlabeled):
    tid = task['id']
    if task.get('total_predictions', 0) > 0:
        preds_resp = requests.get(f'{ls_url}/api/predictions?task={tid}', headers=headers_auth)
        if preds_resp.status_code == 200:
            preds = preds_resp.json()
            if isinstance(preds, dict):
                preds = preds.get('results', [])
            for pred in preds:
                requests.delete(f'{ls_url}/api/predictions/{pred["id"]}', headers=headers_auth)
                deleted += 1
    if (idx + 1) % 50 == 0:
        print(f'  Checked {idx+1}/{len(unlabeled)} tasks for old predictions')
print(f'Deleted {deleted} old predictions')

# Load fine-tuned model
print('Loading fine-tuned model...')
model = YOLO(model_path)

# Build task_id -> filename mapping
task_files = {}
for task in unlabeled:
    url = task['data'].get('image', '')
    filename = url.split('%5C')[-1].split('/')[-1]
    if '%2F' in filename:
        filename = filename.split('%2F')[-1]
    img_path = os.path.join(img_dir, filename)
    if os.path.exists(img_path):
        task_files[task['id']] = img_path

print(f'Running predictions on {len(task_files)} unlabeled images...')

pushed = 0
bird_count = 0
for i, (task_id, img_path) in enumerate(task_files.items()):
    results = model.predict(img_path, conf=0.15, imgsz=1280, verbose=False)
    result = results[0]

    boxes = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
        boxes.append({
            'from_name': 'label',
            'to_name': 'image',
            'type': 'rectanglelabels',
            'value': {
                'x': (x1 / img_w) * 100,
                'y': (y1 / img_h) * 100,
                'width': ((x2 - x1) / img_w) * 100,
                'height': ((y2 - y1) / img_h) * 100,
                'rectanglelabels': [cls_name]
            },
            'score': float(box.conf[0])
        })

    bird_count += len(boxes)

    if boxes:
        pred_body = {
            'task': task_id,
            'result': boxes,
            'score': sum(b['score'] for b in boxes) / len(boxes),
            'model_version': 'bird_v2_finetuned'
        }
        resp = requests.post(f'{ls_url}/api/predictions', headers=headers_json, json=pred_body)
        if resp.status_code == 201:
            pushed += 1

    if (i + 1) % 20 == 0:
        print(f'  Processed {i+1}/{len(task_files)}, {bird_count} detections, {pushed} pushed')

print(f'Done! {bird_count} detections across {len(task_files)} images, {pushed} predictions pushed')
