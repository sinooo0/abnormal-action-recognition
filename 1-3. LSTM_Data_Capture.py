import cv2
import os
import csv
from ultralytics import YOLO

def capture_yolo_pose_frames(video_dir, output_dir, model_path, class_id, fps_capture=3):
    model = YOLO(model_path)
    
    for video_file in os.listdir(video_dir):
        if not video_file.endswith(('.mp4', '.avi', '.mov')):
            continue
        
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        folder_name = f"{class_id}. {video_name}"
        image_dir = os.path.join(output_dir, folder_name, 'images')
        label_dir = os.path.join(output_dir, folder_name, 'labels')
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                results = model(frame)
                annotated_frame = results[0].plot()
                
                frame_filename = os.path.join(image_dir, f"{video_name}_{saved_count}.png")
                cv2.imwrite(frame_filename, annotated_frame)
                
                if results[0].keypoints:
                    keypoints = results[0].keypoints.xy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else range(len(keypoints))
                    
                    for person_id, person_keypoints in zip(ids, keypoints):
                        flat_keypoints = person_keypoints.flatten().tolist()[:34]
                        flat_keypoints += [class_id]
                        
                        csv_filename = os.path.join(label_dir, f"{video_name}_{saved_count}_id_{int(person_id)}.csv")
                        with open(csv_filename, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            headers = [f'kp_{i+1}' for i in range(34)] + ['action_class']
                            writer.writerow(headers)
                            writer.writerow(flat_keypoints)
                
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"{video_file}의 YOLO-pose 프레임 및 키포인트 데이터 프레임별로 저장 완료.")

# 예시 실행
video_dir = './Video/2.theft/'
output_dir = './Capture'
model_path = './Model/yolo11m-pose.pt'
class_id = 2

capture_yolo_pose_frames(video_dir, output_dir, model_path, class_id)
