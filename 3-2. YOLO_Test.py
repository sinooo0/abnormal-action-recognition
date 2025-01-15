import threading
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# 모델 경로 정의
MODEL_PATHS = ["./Model/yolo11m-pose.pt"]
VIDEO_SOURCE = 0  # 웹캠 사용

# 원하는 최대 프레임 크기
MAX_WIDTH = 640
MAX_HEIGHT = 480

# 결과 저장을 위한 리스트
annotated_frames = [None] * len(MODEL_PATHS)
keypoints_data = []
lock = threading.Lock()

def process_frame(model_idx, model, frame, frame_idx):
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()
    detections = results[0].keypoints  # 키포인트 데이터 가져오기
    track_ids = results[0].boxes.id  # 추적된 객체의 ID 가져오기

    # 키포인트 데이터를 ID와 함께 저장
    if detections is not None and track_ids is not None:
        for kp, track_id in zip(detections, track_ids):
            if kp is not None and track_id is not None:
                # numpy 배열로 변환하여 평면화
                flat_keypoints = kp.xy.numpy().flatten().tolist()  # 키포인트를 1D 배열로 변환
                with lock:
                    # ID, frame_idx, keypoints 저장
                    keypoints_data.append([track_id.item(), frame_idx] + flat_keypoints)

    # 비율에 맞게 크기 조정
    original_height, original_width = annotated_frame.shape[:2]
    scale = min(MAX_WIDTH / original_width, MAX_HEIGHT / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

    # 결과 저장 (스레드 안전성 보장)
    with lock:
        annotated_frames[model_idx] = resized_frame

def merge_frames(frames):
    merged_frame = np.zeros_like(frames[0])

    for frame in frames:
        if frame is not None:
            # 각 프레임을 병합 (픽셀 값 덮어쓰기)
            merged_frame = np.maximum(merged_frame, frame)

    return merged_frame

def save_to_csv():
    # 데이터 프레임 생성 및 저장
    columns = ["ID", "Frame"] + [f"kp_{i}" for i in range(1, 35)]  # 34개의 키포인트 (x, y, conf)
    df = pd.DataFrame(keypoints_data, columns=columns)
    df.to_csv("keypoints.csv", index=False)
    print("Keypoints saved to keypoints.csv")

def webcam_capture():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    # YOLO 모델 로드
    models = [YOLO(model_path) for model_path in MODEL_PATHS]
    frame_idx = 0  # 프레임 인덱스 초기화

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Cannot read frame")
            break

        # 각 모델에 대해 스레드 생성하여 프레임 처리
        threads = []
        for idx, model in enumerate(models):
            thread = threading.Thread(
                target=process_frame, 
                args=(idx, model, frame, frame_idx), 
                daemon=True
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # 모든 프레임이 준비되었는지 확인
        with lock:
            if all(frame is not None for frame in annotated_frames):
                # 주석된 프레임을 병합
                combined_frame = merge_frames(annotated_frames)

                # 결과 출력
                cv2.imshow("YOLO Tracking - Combined", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1  # 프레임 인덱스 증가

    cap.release()
    save_to_csv()
    cv2.destroyAllWindows()

# 메인 실행
if __name__ == "__main__":
    webcam_capture()
