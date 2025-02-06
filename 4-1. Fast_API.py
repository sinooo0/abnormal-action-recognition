import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import tensorflow as tf

app = FastAPI()

LSTM_SEQ_LENGTH = 3  # LSTM 시퀀스 수
YOLO_PROCESS_FPS = 3  # 초당 YOLO 프레임 수

# TensorFlow GPU 메모리 증가 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Torch 장치 선택
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 모델 로드 (YOLO Pose, YOLO Weapon, LSTM)
yolo_pose = YOLO("./Model/yolo11s-pose.pt").to(device)
yolo_weapon = YOLO("./Model/yolo11m-weapon.pt").to(device)
lstm_model = load_model("./Model/LSTM.h5", compile=False)
weapon_class_names = yolo_weapon.model.names

# 행동 라벨
action_labels = {0: "Normal", 1: "Doubt", 2: "Danger"}

# 데이터 저장소
object_sequences = {}
previous_actions = {}
previous_accuracies = {}
detected_weapons = []

# 프레임 저장 및 스레드 동기화
latest_frame = None
lock = threading.Lock()

@app.post("/webcam")
async def upload_frame(file: UploadFile = File(...)):
    global latest_frame
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"message": "Failed to decode image"}
    with lock:
        latest_frame = frame
    return {"message": "Frame received"}

def count_valid_keypoints(keypoints_data):
    valid_keypoint_counts = {}
    for obj_id, keypoints in keypoints_data:
        valid_count = sum(1 for kp in keypoints if kp >= 0)
        valid_keypoint_counts[obj_id] = valid_count
    return valid_keypoint_counts

def extract_keypoints(results):
    keypoints_data = []
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(boxes))
        for kp, box, obj_id in zip(keypoints, boxes, ids):
            box_x, box_y, box_w, box_h = box
            relative_keypoints = np.concatenate([
                (kp[:, 0] - (box_x - box_w / 2)) / box_w,
                (kp[:, 1] - (box_y - box_h / 2)) / box_h
            ])
            keypoints_data.append((int(obj_id), relative_keypoints.astype(np.float32)))
    return keypoints_data

def predict_action(obj_id, sequence):
    input_data = np.array(sequence, dtype=np.float32).reshape(1, LSTM_SEQ_LENGTH, -1)
    prediction = lstm_model.predict(input_data, verbose=0)
    previous_actions[obj_id] = int(np.argmax(prediction))
    previous_accuracies[obj_id] = float(np.max(prediction)) * 100

def detect_weapons(frame):
    with torch.no_grad():
        results = yolo_weapon(frame, verbose=False)
    detected_weapons.clear()
    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                              results[0].boxes.cls.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy()):
        detected_weapons.append((tuple(map(int, box)), int(cls), float(conf) * 100))

def process_video():
    global latest_frame
    last_yolo_time = 0
    executor = ThreadPoolExecutor(max_workers=3)
    results = None

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        current_time = time.time()
        if current_time - last_yolo_time >= 1.0 / YOLO_PROCESS_FPS:
            last_yolo_time = current_time
            with torch.no_grad():
                results = yolo_pose.track(frame, persist=True, verbose=False)
            keypoints_data = extract_keypoints(results)
            valid_keypoints = count_valid_keypoints(keypoints_data)

            for obj_id, keypoints in keypoints_data:
                if valid_keypoints.get(obj_id, 0) >= 26:  # 유효한 키포인트 개수 검사
                    if obj_id not in object_sequences:
                        object_sequences[obj_id] = deque(maxlen=LSTM_SEQ_LENGTH)
                    object_sequences[obj_id].append(keypoints)
                    if len(object_sequences[obj_id]) == LSTM_SEQ_LENGTH:
                        executor.submit(predict_action, obj_id, list(object_sequences[obj_id]))

            executor.submit(detect_weapons, frame)

        if results is not None and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(boxes))

            for obj_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                if valid_keypoints.get(obj_id, 0) >= 26:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                    action_label = action_labels.get(previous_actions.get(obj_id, 0), "Normal")
                    accuracy = previous_accuracies.get(obj_id, 0.0)
                    label_text = f"ID {obj_id}: {action_label} ({accuracy:.1f}%)"

                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label_text) * 10, y1), (255, 200, 0), -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for (x1, y1, x2, y2), cls, conf in detected_weapons:
            weapon_label = f"{weapon_class_names[cls]} ({conf:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text_size = cv2.getTextSize(weapon_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1 + 4), (255, 0, 0), -1)
            cv2.putText(frame, weapon_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/predict/")
def predict():
    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
