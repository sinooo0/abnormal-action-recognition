import cv2
import os
import csv
from ultralytics import YOLO

def capture_yolo_pose_frames_with_tracking(video_dir, output_dir, model_path, fps_capture=3):
    model = YOLO(model_path)

    # 하위 폴더 탐색
    for root, dirs, files in os.walk(video_dir):
        for video_file in files:
            if not video_file.endswith(('.mp4', '.avi', '.mov')):
                continue

            # 클래스 ID 추출 (폴더명 맨 앞 숫자)
            relative_path = os.path.relpath(root, video_dir)
            folder_name = os.path.basename(root)
            class_id = int(folder_name.split('.')[0]) if '.' in folder_name else 0

            video_path = os.path.join(root, video_file)
            video_name = os.path.splitext(video_file)[0]

            # 루트 디렉토리 구조를 유지하도록 경로 설정
            output_folder_path = os.path.join(output_dir, relative_path, video_name)
            image_dir = os.path.join(output_folder_path, 'images')
            label_dir = os.path.join(output_folder_path, 'labels')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"비디오 파일을 열 수 없습니다: {video_file}")
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / fps_capture)

            frame_count = 0
            saved_count = 1
            tracker_initialized = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if not tracker_initialized:
                        results = model.track(frame, persist=False)
                        tracker_initialized = True
                    else:
                        results = model.track(frame, persist=True)

                    annotated_frame = results[0].plot()
                    frame_filename = os.path.join(image_dir, f"{video_name}_{saved_count}.png")
                    cv2.imwrite(frame_filename, annotated_frame)

                    if results[0].keypoints:
                        keypoints = results[0].keypoints.xy.cpu().numpy()
                        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(keypoints))

                        for person_id, person_keypoints in zip(ids, keypoints):
                            flat_keypoints = person_keypoints.flatten().tolist()[:34]
                            flat_keypoints += [class_id]

                            person_label_dir = os.path.join(label_dir, f"id_{int(person_id)}")
                            os.makedirs(person_label_dir, exist_ok=True)

                            csv_filename = os.path.join(person_label_dir, f"{video_name}_{saved_count}.csv")
                            with open(csv_filename, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                headers = [f'kp{i}_x' if j % 2 == 0 else f'kp{i}_y' for i in range(17) for j in range(2)] + ['action_class']
                                writer.writerow(headers)
                                writer.writerow(flat_keypoints)

                    saved_count += 1

                frame_count += 1

            cap.release()
            print(f"{video_file}의 YOLO-pose 프레임 및 키포인트 데이터 저장 완료.")

# 실행
video_dir = './Video/'
output_dir = './Data/LSTM_Capture'
model_path = './Model/yolo11m-pose.pt'

capture_yolo_pose_frames_with_tracking(video_dir, output_dir, model_path)
