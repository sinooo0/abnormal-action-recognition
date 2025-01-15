import os
import cv2
import pandas as pd
from ultralytics import YOLO
import logging

# YOLO 로그 비활성화
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# YOLO Pose 결과에서 키포인트만 추출
def extract_keypoints(results):
    keypoints_data = []
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        detections = results[0].keypoints
        ids = results[0].boxes.id.cpu().numpy().tolist()
        for kp, obj_id in zip(detections, ids):
            if kp is not None:
                flat_keypoints = kp.xy.cpu().numpy().flatten().tolist()
                keypoints_data.append((obj_id, flat_keypoints))
    return keypoints_data

# 중복 파일명이 있으면 숫자를 증가시켜 저장
def get_unique_filename(output_dir, file_name):
    file_path = os.path.join(output_dir, f"{file_name}_1.csv")
    counter = 2
    while os.path.exists(file_path):
        file_path = os.path.join(output_dir, f"{file_name}_{counter}.csv")
        counter += 1
    return file_path

# 실시간 웹캠에서 키포인트 추출 및 CSV 저장
def process_webcam(yolo_model_path, action_class, file_name, camera_index=0, target_fps=2, max_frames=30, output_base_dir="./LSTM_Data"):
    model = YOLO(yolo_model_path)
    model.overrides['verbose'] = False  # YOLO 내부 로그 출력 비활성화
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: 웹캠 {camera_index}을(를) 열 수 없습니다.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(original_fps / target_fps), 1)
    print(f"웹캠 {camera_index} 시작 | Original FPS: {original_fps}, Target FPS: {target_fps}, Frame Interval: {frame_interval}")

    # CSV 저장 경로 설정 (동적으로 폴더 생성)
    action_dir = os.path.join(output_base_dir, str(action_class))
    os.makedirs(action_dir, exist_ok=True)

    keypoints_data = {}
    saved_ids = set()
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: 웹캠 {camera_index}에서 프레임을 읽을 수 없습니다.")
            break

        if frame_idx % frame_interval == 0:
            results = model.track(frame, persist=True)
            frame_keypoints = extract_keypoints(results)
            for obj_id, keypoints in frame_keypoints:
                keypoints.append(action_class)
                if obj_id not in keypoints_data:
                    keypoints_data[obj_id] = []
                
                keypoints_data[obj_id].append(keypoints)

                if len(keypoints_data[obj_id]) >= max_frames and obj_id not in saved_ids:
                    # ID별 폴더 생성
                    person_dir = os.path.join(action_dir, f"id_{int(obj_id)}")
                    os.makedirs(person_dir, exist_ok=True)

                    # 파일 저장 경로 설정
                    output_csv = get_unique_filename(person_dir, file_name)

                    # CSV 헤더 수정: kp0_x, kp0_y, ..., kp16_x, kp16_y, action_class
                    columns = [f"kp{i}_x" if j % 2 == 0 else f"kp{i}_y" for i in range(17) for j in range(2)] + ["action_class"]
                    df = pd.DataFrame(keypoints_data[obj_id], columns=columns)
                    df.to_csv(output_csv, index=False)
                    print(f"ID {obj_id} 데이터가 {output_csv}에 저장되었습니다.")

                    saved_ids.add(obj_id)
                    del keypoints_data[obj_id]

            # 모든 ID가 max_frames만큼 저장되었는지 확인 후 종료
            detected_ids = {obj_id for obj_id, _ in frame_keypoints}
            if detected_ids.issubset(saved_ids):
                print("모든 ID 데이터가 저장되어 프로그램을 종료합니다.")
                cap.release()
                cv2.destroyAllWindows()
                return

            # 결과 시각화
            annotated_frame = results[0].plot()
            cv2.imshow(f"YOLO Pose Detection (Camera {camera_index})", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    yolo_model_path = "./Model/yolo11m-pose.pt"  # YOLO Pose 모델 경로
    target_fps = 3  # 초당 처리할 프레임 수
    max_frames = 3  # 저장할 최대 프레임 수

    camera_index = 0  # 사용할 웹캠 번호
    action_class = 5  # 행동 클래스
    file_name = "scan"  # 저장할 파일 이름 (확장자 제외)
    output_base_dir = "./Data/LSTM_Live"  # 데이터 저장 경로

    process_webcam(yolo_model_path, action_class, file_name, camera_index, target_fps, max_frames, output_base_dir)
