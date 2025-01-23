import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import threading
import time

# YOLO Pose 결과에서 키포인트를 추출하는 함수 (상대 좌표 계산 포함)
def extract_keypoints(results):
    keypoints_data = []
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()  # 키포인트 좌표
        boxes = results[0].boxes.xywh.cpu().numpy()       # 바운딩 박스 [x_center, y_center, width, height]
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(keypoints))

        for kp, box, obj_id in zip(keypoints, boxes, ids):
            box_x, box_y, box_w, box_h = box
            relative_keypoints = []
            for i in range(17):  # 17개의 키포인트 처리 (COCO 포맷 기준)
                kp_x = (kp[i, 0] - (box_x - box_w / 2)) / box_w  # x 상대 좌표
                kp_y = (kp[i, 1] - (box_y - box_h / 2)) / box_h  # y 상대 좌표
                relative_keypoints.extend([kp_x, kp_y])
            keypoints_data.append((obj_id, relative_keypoints))  # ID와 상대 좌표 저장
    return keypoints_data

# 프레임 크기를 재조정하는 함수 (최대 너비와 높이를 기준으로 조정)
def resize_frame(frame, max_width=640, max_height=480):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width, new_height = int(width * scale), int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

# 행동 클래스 라벨 매핑
action_labels = {0: "Normal", 1: "Doubt", 2: "Danger"}

# LSTM을 사용하여 행동을 예측하는 함수 (멀티스레딩 활용)
def predict_action(obj_id, sequence, lstm_model, seq_length, previous_actions, previous_accuracies):
    input_data = np.array(sequence).reshape(1, seq_length, -1)
    prediction = lstm_model.predict(input_data, verbose=0)
    action_class = np.argmax(prediction)
    accuracy = float(np.max(prediction)) * 100
    previous_actions[obj_id] = action_class
    previous_accuracies[obj_id] = accuracy

# YOLO와 LSTM을 사용하여 비디오나 웹캠을 처리하는 함수
def process_video_or_webcam(yolo_model_path, lstm_model_path, seq_length, target_fps=5, video_path=None, camera_index=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_model = YOLO(yolo_model_path).to(device)
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
    retention_time = 0.5

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

        for obj_id, keypoints in keypoints_data:
            if obj_id not in object_sequences:
                object_sequences[obj_id] = deque(maxlen=seq_length)
            object_sequences[obj_id].append(keypoints)

            if len(object_sequences[obj_id]) == seq_length:
                threading.Thread(
                    target=predict_action,
                    args=(obj_id, object_sequences[obj_id], lstm_model, seq_length, previous_actions, previous_accuracies)
                ).start()

        for obj_id, box in zip(results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [], results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            previous_boxes[obj_id] = box
            last_seen[obj_id] = current_time

            action_label = action_labels.get(previous_actions.get(obj_id), "Normal")
            accuracy = previous_accuracies.get(obj_id, 0.0)
            label_text = f"ID {obj_id}: {action_label} ({accuracy:.1f}%)"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("YOLO Pose + LSTM Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Processing stopped.")

if __name__ == "__main__":
    yolo_model_path = "./Model/yolo11s-pose.pt"
    lstm_model_path = "./Model/LSTM.h5"
    seq_length = 3
    target_fps = 3
    video_path = None
    camera_index = 1

    process_video_or_webcam(yolo_model_path, lstm_model_path, seq_length, target_fps, video_path, camera_index)
