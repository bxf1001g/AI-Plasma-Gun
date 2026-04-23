"""Upload new DJI_0578 frames to Label Studio via local storage and auto-label with bird_v3."""
import requests
import os
import re
import time
from urllib.parse import unquote
from ultralytics import YOLO

token = '5a32889a9a833d4186ba64b2632bc221ee0ca83e'
ls_url = 'http://localhost:8080'
headers_json = {'Authorization': f'Token {token}', 'Content-Type': 'application/json'}
headers_auth = {'Authorization': f'Token {token}'}
img_dir = r'D:\datasets\birds\frames_dji_0578'
model_path = r'D:\datasets\birds\runs\bird_v3\weights\best.pt'

# Step 1: Create project
label_config = (
    '<View>'
    '  <Image name="image" value="$image"/>'
    '  <RectangleLabels name="label" toName="image">'
    '    <Label value="bird" background="#FF0000"/>'
    '    <Label value="idle_bird" background="#00FF00"/>'
    '  </RectangleLabels>'
    '</View>'
)

project_data = {
    'title': 'Bird Auto-Label v3 (DJI_0578)',
    'description': 'Auto-labeling 2218 frames from DJI_0578 with bird_v3 model',
    'label_config': label_config
}
resp = requests.post(f'{ls_url}/api/projects', headers=headers_json, json=project_data)
print(f'Create project: {resp.status_code}')
project = resp.json()
pid = project['id']
print(f'Project ID: {pid}')

# Step 2: Add local file storage (same approach as project 2)
storage_data = {
    'project': pid,
    'title': 'DJI_0578 Frames',
    'path': img_dir,
    'regex_filter': '.*\\.jpg',
    'use_blob_urls': True,
    'recursive_scan': False
}
resp = requests.post(f'{ls_url}/api/storages/localfiles', headers=headers_json, json=storage_data)
print(f'Create storage: {resp.status_code}')
storage = resp.json()
storage_id = storage['id']
print(f'Storage ID: {storage_id}')

# Step 3: Sync storage to create tasks
print('Syncing storage...')
resp = requests.post(f'{ls_url}/api/storages/localfiles/{storage_id}/sync', headers=headers_auth)
print(f'Sync started: {resp.status_code}')

# Wait for sync to complete
for attempt in range(60):
    time.sleep(2)
    resp = requests.get(f'{ls_url}/api/storages/localfiles/{storage_id}', headers=headers_auth)
    info = resp.json()
    status = info.get('status', '')
    count = info.get('last_sync_count')
    if status == 'completed':
        print(f'Sync completed: {count} tasks created')
        break
    elif status == 'failed':
        print(f'Sync failed: {info.get("traceback", "unknown error")}')
        exit(1)
    if attempt % 5 == 0:
        print(f'  Waiting... status={status}')
else:
    print('Sync timeout after 120s')
    exit(1)

# Step 4: Fetch all tasks with pagination
print(f'\nFetching tasks from project {pid}...')
all_tasks = []
page = 1
while True:
    resp = requests.get(
        f'{ls_url}/api/projects/{pid}/tasks',
        headers=headers_auth,
        params={'page_size': 100, 'page': page}
    )
    data = resp.json()
    tasks = data if isinstance(data, list) else data.get('results', [])
    all_tasks.extend(tasks)
    if len(tasks) < 100:
        break
    page += 1
print(f'Total tasks: {len(all_tasks)}')

# Step 5: Run YOLO predictions
print(f'\nLoading bird_v3 model...')
model = YOLO(model_path)

# Build task_id -> local image path mapping
# Local storage URLs look like: /data/local-files/?d=frames_dji_0578\filename.jpg
task_files = {}
for task in all_tasks:
    url = task['data'].get('image', '')
    # Extract the relative path from the URL query param
    if 'd=' in url:
        rel_path = url.split('d=')[-1]
        rel_path = unquote(rel_path)
        # Document root is D:\datasets\birds, so full path = root + rel_path
        full_path = os.path.join(r'D:\datasets\birds', rel_path)
        if os.path.exists(full_path):
            task_files[task['id']] = full_path

print(f'Running predictions on {len(task_files)} images...')

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
            'model_version': 'bird_v3_finetuned'
        }
        resp = requests.post(f'{ls_url}/api/predictions', headers=headers_json, json=pred_body)
        if resp.status_code == 201:
            pushed += 1

    if (i + 1) % 100 == 0:
        print(f'  Processed {i+1}/{len(task_files)}, {bird_count} detections, {pushed} pushed')

print(f'\nDone! {bird_count} detections across {len(task_files)} images, {pushed} predictions pushed')
print(f'Project {pid} ready for review in Label Studio')
