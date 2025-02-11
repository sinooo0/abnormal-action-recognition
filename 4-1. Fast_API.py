import os
import cv2
import time
import torch
import uvicorn
import requests
import threading
import numpy as np
from queue import Queue
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from fastapi.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf

app = FastAPI()

SPRING_URL = "https://crime-spring-h4hwhacpa6a2f0d0.koreacentral-01.azurewebsites.net/api/anomalies"
FACE_API_URL = "https://face-api-dnbbgjgmh6gvdug7.koreacentral-01.azurewebsites.net/detect"

LSTM_SEQ_LENGTH = 6  # LSTM 시퀀스 수
YOLO_PROCESS_FPS = 6  # 초당 YOLO 프레임 수

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
yolo_pose = YOLO("./Model/yolo11m-pose.pt").to(device)
yolo_weapon = YOLO("./Model/yolo-weapon.pt").to(device)
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
frame_queue = Queue(maxsize=1)
lstm_queue = Queue()

# 중복 알림 전송 방지
last_sent_time = 0
min_alert_interval = 10 # 알림 간격 설정(초)
sent_frames = set()

# 웹캠 프레임 수신
@app.post("/webcam")
async def upload_frame(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"message": "Failed to decode image"}

    if not frame_queue.empty():
        frame_queue.get()

    frame_queue.put(frame)
    return {"message": "Frame received"}


def lstm_worker():
    while True:
        obj_id, sequence = lstm_queue.get()
        predict_action(obj_id, sequence)
        lstm_queue.task_done()

lstm_thread = threading.Thread(target=lstm_worker, daemon=True)
lstm_thread.start()

alert_executor = ThreadPoolExecutor(max_workers=2)

# 유효 키포인트 처리
def count_valid_keypoints(keypoints_data):
    valid_keypoint_counts = {}
    for obj_id, keypoints in keypoints_data:
        valid_count = sum(1 for kp in keypoints if kp >= 0)
        valid_keypoint_counts[obj_id] = valid_count
    return valid_keypoint_counts

# 키포인트 추출
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

# 행동 예측
def predict_action(obj_id, sequence):
    input_data = np.array(sequence, dtype=np.float32).reshape(1, LSTM_SEQ_LENGTH, -1)
    with tf.device('/CPU:0'):
        prediction = lstm_model.predict(input_data, verbose=0)

    action_idx = int(np.argmax(prediction))
    accuracy = float(np.max(prediction)) * 100
    previous_actions[obj_id] = action_idx
    previous_accuracies[obj_id] = accuracy

# 무기 탐지
def detect_weapons(frame):
    with torch.no_grad():
        results = yolo_weapon(frame, conf=0.8, verbose=False)

    detected_weapons.clear()
    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                              results[0].boxes.cls.cpu().numpy(),
                              results[0].boxes.conf.cpu().numpy()):
        confidence = float(conf) * 100
        detected_weapons.append((tuple(map(int, box)), int(cls), confidence))

# 위험 알림 전송
def crime_alert(frame, raw_frame, alert_type, confidence):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{confidence:.1f}_{timestamp}.jpg"

    if filename in sent_frames:
        return
    sent_frames.add(filename)

    # SPRING 서버로 탐지 이미지 전송
    _, buffer = cv2.imencode(".jpg", frame)
    img_bytes = buffer.tobytes()

    files = {'file': (filename, img_bytes, 'image/jpeg')}
    response = requests.post(SPRING_URL, files=files)

    print(f"Sent {filename} to Spring, Response: {response.status_code}")
    
    # FACE_API 서버로 탐지 이미지 전송
    _, buffer = cv2.imencode(".jpg", raw_frame)
    img_bytes = buffer.tobytes()

    files = {'file': (filename, img_bytes, 'image/jpeg')}
    response = requests.post(FACE_API_URL, files=files)

    print(f"Sent RAW {filename} to Face_API, Response: {response.status_code}")

# 알림 조건 설정
def process_alerts(frame, raw_frame, detected_weapons, previous_actions, previous_accuracies):
    global last_sent_time

    # 알림 전송 시간 설정
    current_time = time.time()
    if current_time - last_sent_time < min_alert_interval:
        return

    danger_detected = False
    weapon_detected = False
    highest_weapon_conf = 0
    weapon_class_name = ""

    # Danger가 95% 이상이면 알림
    for obj_id, action in previous_actions.items():
        accuracy = previous_accuracies.get(obj_id, 0.0)
        if action == 2 and accuracy >= 95:
            danger_detected = True
            danger_accuracy = accuracy

    # 무기 탐지가 80% 이상이면 알림
    for (x1, y1, x2, y2), cls, conf in detected_weapons:
        if conf >= 80 and conf > highest_weapon_conf:
            highest_weapon_conf = conf
            weapon_class_name = weapon_class_names[cls]
            weapon_detected = True

    # 동시에 탐지되면 무기를 우선순위로 알림을 전송
    if weapon_detected:
        alert_executor.submit(crime_alert, frame, raw_frame, weapon_class_name, highest_weapon_conf)
        last_sent_time = current_time
    elif danger_detected:
        alert_executor.submit(crime_alert, frame, raw_frame, "Danger", danger_accuracy)
        last_sent_time = current_time

    # 중복 알림 방지
    if weapon_detected or danger_detected:
        reset_time = current_time + min_alert_interval - 1
        threading.Timer(reset_time - current_time, reset_object_state, args=(previous_actions, previous_accuracies)).start()

# 클래스 정보 초기화
def reset_object_state(previous_actions, previous_accuracies):
    previous_actions.clear()
    previous_accuracies.clear()

# 전체 영상 처리
def process_video():
    last_yolo_time = 0
    executor = ThreadPoolExecutor(max_workers=3)
    results = None

    while True:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        raw_frame = frame.copy()

        current_time = time.time()
        if current_time - last_yolo_time >= 1.0 / YOLO_PROCESS_FPS:
            last_yolo_time = current_time
            with torch.no_grad():
                results = yolo_pose.track(frame, persist=True, verbose=False)
            keypoints_data = extract_keypoints(results)
            valid_keypoints = count_valid_keypoints(keypoints_data)

            for obj_id, keypoints in keypoints_data:
                if keypoints is None or len(keypoints) == 0:
                    continue

                if valid_keypoints.get(obj_id, 0) >= 24:
                    if obj_id not in object_sequences:
                        object_sequences[obj_id] = deque(maxlen=LSTM_SEQ_LENGTH)

                    object_sequences[obj_id].append(keypoints)

                    if len(object_sequences[obj_id]) == LSTM_SEQ_LENGTH:
                        lstm_queue.put((obj_id, list(object_sequences[obj_id])))

            executor.submit(detect_weapons, frame)

        if results is not None and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(boxes))

            for obj_id, box in zip(ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                if valid_keypoints.get(obj_id, 0) >= 24:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                    action_label = action_labels.get(previous_actions.get(obj_id, 0), "Normal")
                    accuracy = previous_accuracies.get(obj_id, 100.0)
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

        executor.submit(process_alerts, frame, raw_frame, detected_weapons, previous_actions, previous_accuracies)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/predict")
def predict():
    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
