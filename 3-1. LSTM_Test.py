import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import threading
import time

# YOLO Pose 결과에서 키포인트를 추출하는 함수
def extract_keypoints(results):
    keypoints_data = []
    # 결과에서 keypoints와 box ID가 있는 경우 처리
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        detections = results[0].keypoints
        for kp in detections:
            if kp is not None:
                # 키포인트를 평면 리스트로 변환하여 저장
                flat_keypoints = kp.xy.cpu().numpy().flatten().tolist()
                keypoints_data.append(flat_keypoints)
    return keypoints_data

# 프레임 크기를 재조정하는 함수 (최대 너비와 높이를 기준으로 조정)
def resize_frame(frame, max_width=640, max_height=480):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)  # 축소 비율 계산
    new_width, new_height = int(width * scale), int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

# 행동 클래스 라벨 매핑 (예: Normal, Fight 등)
action_labels = {0: "Normal", 1: "Fight", 2: "Theft", 3: "Unbag", 4: "Scan"}

# LSTM을 사용하여 행동을 예측하는 함수 (멀티스레딩 활용)
def predict_action(obj_id, sequence, lstm_model, seq_length, previous_actions, previous_accuracies):
    # 시퀀스를 모델 입력 형식으로 변환
    input_data = np.array(sequence).reshape(1, seq_length, -1)
    # LSTM 모델로 예측 수행
    prediction = lstm_model.predict(input_data, verbose=0)
    action_class = np.argmax(prediction)  # 가장 높은 확률의 클래스 추출
    accuracy = float(np.max(prediction)) * 100  # 확률을 백분율로 변환
    # 예측 결과 저장
    previous_actions[obj_id] = action_class
    previous_accuracies[obj_id] = accuracy

# YOLO와 LSTM을 사용하여 비디오나 웹캠을 처리하는 함수
def process_video_or_webcam(yolo_model_path, lstm_model_path, seq_length, target_fps=5, video_path=None, camera_index=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU 사용 여부 확인
    yolo_model = YOLO(yolo_model_path).to(device)  # YOLO 모델 로드 및 디바이스 설정
    lstm_model = load_model(lstm_model_path, compile=False)  # LSTM 모델 로드
    
    # 객체 추적 및 상태 저장을 위한 딕셔너리 초기화
    object_sequences = {}  # 객체별 키포인트 시퀀스 저장
    previous_actions = {}  # 이전 행동 예측 저장
    previous_accuracies = {}  # 이전 예측 정확도 저장
    previous_boxes = {}  # 이전 박스 좌표 저장
    last_seen = {}  # 마지막 감지 시간을 저장

    # 비디오 캡처 초기화
    cap = cv2.VideoCapture(camera_index) if video_path is None else cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to access {'webcam' if video_path is None else video_path}.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 원본 비디오 FPS 확인
    frame_interval = int(original_fps / target_fps)  # 처리할 프레임 간격 계산
    frame_idx = 0
    retention_time = 0.5  # 사라진 객체의 정보를 유지할 시간 (초 단위)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = resize_frame(frame)  # 프레임 크기 조정
        current_time = time.time()  # 현재 시간 저장

        # YOLO 모델로 키포인트 및 객체 감지 수행
        with torch.cuda.amp.autocast():  # 자동 혼합 정밀도 활성화
            results = yolo_model.track(frame, persist=True, verbose=False)
        keypoints_data = extract_keypoints(results)  # 키포인트 추출
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []  # 객체 ID 추출

        current_ids = set(ids)  # 현재 프레임의 객체 ID 집합

        # 사라진 객체의 이전 박스를 일정 시간 유지
        for obj_id in list(previous_boxes.keys()):
            if obj_id not in current_ids:
                if current_time - last_seen.get(obj_id, 0) <= retention_time:
                    # 유지 시간 내라면 이전 박스와 행동 정보 표시
                    box = previous_boxes[obj_id]
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    action_label = action_labels.get(previous_actions.get(obj_id), "Normal")
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

        # 현재 감지된 객체 박스를 업데이트
        for box, obj_id in zip(results[0].boxes.xyxy, ids):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            previous_boxes[obj_id] = box.cpu().numpy()
            last_seen[obj_id] = current_time  # 마지막 감지 시간 갱신

            # 행동 정보 표시
            action_label = action_labels.get(previous_actions.get(obj_id), "Normal")
            accuracy = previous_accuracies.get(obj_id, 0.0)
            label_text = f"ID {obj_id}: {action_label} ({accuracy:.1f}%)"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # LSTM 예측 수행
        for obj_id, keypoints in zip(ids, keypoints_data):
            if obj_id not in object_sequences:
                object_sequences[obj_id] = deque(maxlen=seq_length)
            object_sequences[obj_id].append(keypoints)

            if len(object_sequences[obj_id]) == seq_length:
                # 멀티스레딩으로 예측 실행
                threading.Thread(target=predict_action, args=(obj_id, object_sequences[obj_id], lstm_model, seq_length, previous_actions, previous_accuracies)).start()

        # 프레임 출력
        cv2.imshow("YOLO Pose + LSTM Action Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키로 종료
            break

        frame_idx += 1

    cap.release()  # 비디오 캡처 해제
    cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
    print("Processing stopped.")

if __name__ == "__main__":
    # YOLO와 LSTM 모델 경로 및 설정값
    yolo_model_path = "./Model/yolo11s-pose.pt"
    lstm_model_path = "./Model/LSTM.h5"
    seq_length = 3  # LSTM 입력 시퀀스 길이
    target_fps = 3  # 목표 FPS
    video_path = None  # 비디오 경로 (None이면 웹캠 사용)
    camera_index = 1  # 웹캠 인덱스

    # 처리 시작
    process_video_or_webcam(yolo_model_path, lstm_model_path, seq_length, target_fps, video_path, camera_index)
