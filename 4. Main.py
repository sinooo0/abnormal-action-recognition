import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# 🔹 행동 라벨 (LSTM)
action_labels = {0: "Normal", 1: "Doubt", 2: "Danger"}

# 🔹 YOLO Pose 키포인트 추출
def extract_keypoints(results):
    keypoints_data = []
    if results[0].keypoints is not None and results[0].boxes is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(keypoints))

        for kp, box, obj_id in zip(keypoints, boxes, ids):
            box_x, box_y, box_w, box_h = box
            relative_keypoints = np.concatenate([(kp[:, 0] - (box_x - box_w / 2)) / box_w, 
                                                 (kp[:, 1] - (box_y - box_h / 2)) / box_h])
            keypoints_data.append((obj_id, relative_keypoints.astype(np.float32)))
    return keypoints_data

# 🔹 LSTM 행동 예측
def predict_action(obj_id, sequence, lstm_model, seq_length, previous_actions, previous_accuracies):
    input_data = np.array(sequence, dtype=np.float32).reshape(1, seq_length, -1)
    prediction = lstm_model.predict(input_data, verbose=0)
    previous_actions[obj_id] = int(np.argmax(prediction))
    previous_accuracies[obj_id] = float(np.max(prediction)) * 100

# 🔹 YOLO Object Detection (흉기 탐지)
def detect_weapons(yolo_weapon, frame, detected_weapons):
    with torch.no_grad():
        results = yolo_weapon(frame, verbose=False)
    detected_weapons.clear()
    for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
        detected_weapons.append((tuple(map(int, box)), int(cls), float(conf) * 100))

# 🔹 메인 실행 함수 (YOLO Pose + LSTM + YOLO Weapon Detection)
def process_video_or_webcam(yolo_pose_path, yolo_weapon_path, lstm_model_path, seq_length, target_fps=5, video_path=None, camera_index=0):
    device = 'cpu'
    torch.set_num_threads(4)
    yolo_pose = YOLO(yolo_pose_path).to(device)
    yolo_weapon = YOLO(yolo_weapon_path).to(device)
    lstm_model = load_model(lstm_model_path, compile=False)

    weapon_class_names = yolo_weapon.model.names

    object_sequences = {}
    previous_actions = {}
    previous_accuracies = {}
    detected_weapons = []

    cap = cv2.VideoCapture(camera_index) if video_path is None else cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to access {'webcam' if video_path is None else video_path}.")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(original_fps / target_fps))
    frame_idx = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # 🔹 YOLO Pose 추적
            if frame_idx % frame_interval == 0:
                with torch.no_grad():
                    results = yolo_pose.track(frame, persist=True, verbose=False)
                keypoints_data = extract_keypoints(results)

                for obj_id, keypoints in keypoints_data:
                    if obj_id not in object_sequences:
                        object_sequences[obj_id] = deque(maxlen=seq_length)
                    object_sequences[obj_id].append(keypoints)

                    if len(object_sequences[obj_id]) == seq_length:
                        executor.submit(predict_action, obj_id, list(object_sequences[obj_id]), lstm_model, seq_length, previous_actions, previous_accuracies)

            # 🔹 YOLO Object Detection (흉기 탐지)
            if frame_idx % (frame_interval * 3) == 0:
                executor.submit(detect_weapons, yolo_weapon, frame, detected_weapons)

            # 🔹 행동 인식 결과 (클래스명 + 신뢰도 출력)
            for obj_id, box in zip(results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [], results[0].boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)

                action_label = action_labels.get(previous_actions.get(obj_id, 0), "Normal")
                accuracy = previous_accuracies.get(obj_id, 0.0)
                label_text = f"ID {obj_id}: {action_label} ({accuracy:.1f}%)"

                cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label_text) * 10, y1), (255, 200, 0), -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 🔹 흉기 탐지 결과 (클래스명 + 신뢰도 출력)
            for (x1, y1, x2, y2), cls_id, conf in detected_weapons:
                label = f"{weapon_class_names.get(cls_id, 'Unknown')} ({conf:.1f}%)"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = x1, y1 - 10
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 4), (text_x + text_size[0], text_y + 4), (255, 0, 0), -1)
                cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("YOLO Pose + LSTM Action Recognition + YOLO Weapon Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Processing stopped.")


if __name__ == "__main__":
    yolo_pose_path = "./Model/yolo11s-pose.pt"
    yolo_weapon_path = "./Model/yolo11n-weapon.pt"
    lstm_model_path = "./Model/LSTM.h5"
    seq_length = 3
    target_fps = 5
    video_path = None
    camera_index = 0

    process_video_or_webcam(yolo_pose_path, yolo_weapon_path, lstm_model_path, seq_length, target_fps, video_path, camera_index)
