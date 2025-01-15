import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import threading
import time

# YOLO Pose 결과에서 키포인트 추출
def extract_keypoints(results):
    keypoints_data = []
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        detections = results[0].keypoints
        for kp in detections:
            if kp is not None:
                flat_keypoints = kp.xy.cpu().numpy().flatten().tolist()
                keypoints_data.append(flat_keypoints)
    return keypoints_data

# 프레임 크기 조정
def resize_frame(frame, max_width=640, max_height=480):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width, new_height = int(width * scale), int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

# 행동 클래스 매핑
action_labels = {0: "Normal", 1: "Fight", 2: "Theft", 3:"Unbag", 4: "Scan"}

# LSTM 예측 함수 (멀티스레딩)
def predict_action(obj_id, sequence, lstm_model, seq_length, previous_actions, previous_accuracies):
    input_data = np.array(sequence).reshape(1, seq_length, -1)
    prediction = lstm_model.predict(input_data, verbose=0)
    action_class = np.argmax(prediction)
    accuracy = float(np.max(prediction)) * 100
    previous_actions[obj_id] = action_class
    previous_accuracies[obj_id] = accuracy

# 실시간 처리 및 최적화
def process_video_or_webcam(yolo_model_path, lstm_model_path, seq_length, target_fps=5, video_path=None, camera_index=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO(yolo_model_path)
    yolo_model.to(device)
    lstm_model = load_model(lstm_model_path, compile=False)
    
    object_sequences = {}
    previous_actions = {}
    previous_accuracies = {}
    previous_boxes = {}
    last_seen = {}

    cap = cv2.VideoCapture(camera_index) if video_path is None else cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to access {'webcam' if video_path is None else video_path}.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(original_fps / target_fps)
    frame_idx = 0
    retention_time = 0.5  # 0.5초 유지

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = resize_frame(frame)
        current_time = time.time()

        with torch.cuda.amp.autocast():
            results = yolo_model.track(frame, persist=True, verbose=False)
        keypoints_data = extract_keypoints(results)
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

        current_ids = set(ids)

        # 사라진 객체의 이전 박스를 0.5초 동안 유지
        for obj_id in list(previous_boxes.keys()):
            if obj_id not in current_ids:
                if current_time - last_seen.get(obj_id, 0) <= retention_time:
                    # 유지 시간 내라면 이전 박스 그리기
                    box = previous_boxes[obj_id]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    action_label = action_labels.get(previous_actions.get(obj_id), "Walk")
                    accuracy = previous_accuracies.get(obj_id, 0.0)
                    label_text = f"ID {obj_id}: {action_label} ({accuracy:.1f}%)"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    # 유지 시간이 지나면 데이터 삭제
                    previous_boxes.pop(obj_id, None)
                    object_sequences.pop(obj_id, None)
                    previous_actions.pop(obj_id, None)
                    previous_accuracies.pop(obj_id, None)
                    last_seen.pop(obj_id, None)

        # 현재 감지된 객체 박스 업데이트
        for box, obj_id in zip(results[0].boxes.xyxy, ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            previous_boxes[obj_id] = box.cpu().numpy()
            last_seen[obj_id] = current_time  # 마지막 감지 시간 업데이트

            action_label = action_labels.get(previous_actions.get(obj_id), "Walk")
            accuracy = previous_accuracies.get(obj_id, 0.0)
            label_text = f"ID {obj_id}: {action_label} ({accuracy:.1f}%)"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # LSTM 예측 수행
        for obj_id, keypoints in zip(ids, keypoints_data):
            if obj_id not in object_sequences:
                object_sequences[obj_id] = deque(maxlen=seq_length)
            object_sequences[obj_id].append(keypoints)

            if len(object_sequences[obj_id]) == seq_length:
                threading.Thread(target=predict_action, args=(obj_id, object_sequences[obj_id], lstm_model, seq_length, previous_actions, previous_accuracies)).start()

        cv2.imshow("YOLO Pose + LSTM Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Processing stopped.")

if __name__ == "__main__":
    yolo_model_path = "./Model/yolo11m-pose.pt"
    lstm_model_path = "./Model/LSTM.h5"
    seq_length = 3
    target_fps = 5
    video_path = None
    camera_index = 2

    process_video_or_webcam(yolo_model_path, lstm_model_path, seq_length, target_fps, video_path, camera_index)
