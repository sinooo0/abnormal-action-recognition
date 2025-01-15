import os
import cv2
import pandas as pd
from ultralytics import YOLO
import re

# YOLO Pose 결과에서 키포인트만 추출
def extract_keypoints(results):
    keypoints_data = []
    if results[0].keypoints is not None and results[0].boxes.id is not None:
        detections = results[0].keypoints
        for kp in detections:
            if kp is not None:
                flat_keypoints = kp.xy.cpu().numpy().flatten().tolist()
                keypoints_data.append(flat_keypoints)
    return keypoints_data

# 초당 target_fps 프레임씩 키포인트 데이터를 추출하고 CSV로 저장
def process_video_to_single_csv(video_path, yolo_model_path, output_csv, target_fps=2, action_class=0, max_frames=6):
    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} 파일을 열 수 없습니다.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(original_fps / target_fps), 1)
    print(f"Processing: {video_path} | Original FPS: {original_fps}, Target FPS: {target_fps}, Frame Interval: {frame_interval}")

    keypoints_data = []
    frame_idx = 0

    while len(keypoints_data) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model.track(frame, persist=True)
            frame_keypoints = extract_keypoints(results)
            for keypoints in frame_keypoints:
                keypoints.append(action_class)
                keypoints_data.append(keypoints)
                if len(keypoints_data) >= max_frames:
                    break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if not keypoints_data:
        print(f"경고: {video_path}에서 키포인트 데이터가 없습니다.")
        return

    columns = [f"kp_{i}" for i in range(1, 35)] + ["action_class"]
    df = pd.DataFrame(keypoints_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"키포인트 데이터와 행동 클래스가 {output_csv}에 저장되었습니다.")

# 폴더 내 모든 동영상을 처리 (폴더 이름의 첫 숫자만 추출)
def process_video_folder(folder_path, yolo_model_path, output_folder, target_fps=2, max_frames=6):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_path):
            continue  # 폴더가 아니면 건너뜀

        # 폴더 이름에서 첫 번째 숫자 추출 (예: "0. Run" → 0)
        match = re.match(r"(\d+)", class_folder)
        if match:
            action_class = int(match.group(1))
        else:
            print(f"경고: '{class_folder}'에서 숫자를 찾을 수 없습니다. 건너뜁니다.")
            continue

        class_output_folder = os.path.join(output_folder, str(action_class))
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)

        video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            output_csv = os.path.join(class_output_folder, os.path.splitext(video_file)[0] + '.csv')
            process_video_to_single_csv(video_path, yolo_model_path, output_csv, target_fps, action_class, max_frames)

if __name__ == "__main__":
    video_folder_path = "./Video"  # 동영상 파일이 저장된 폴더 경로
    yolo_model_path = "./Model/yolo11m-pose.pt"  # YOLO Pose 모델 경로
    output_folder = "./LSTM_Data1"  # CSV 파일을 저장할 폴더 경로
    target_fps = 3  # 초당 처리할 프레임 수
    max_frames = 15  # 처리할 최대 프레임 수

    process_video_folder(video_folder_path, yolo_model_path, output_folder, target_fps, max_frames)
