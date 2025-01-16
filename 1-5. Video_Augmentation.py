import cv2
import os
import numpy as np

def get_next_filename(directory, base_name, extension='.mp4'):
    os.makedirs(directory, exist_ok=True)
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    numbers = [int(f[len(base_name):-len(extension)]) for f in existing_files if f[len(base_name):-len(extension)].isdigit()]
    next_number = max(numbers, default=0) + 1
    return os.path.join(directory, f"{base_name}{next_number}{extension}")

def augment_video(video_path, save_dir, speed=1.0, rotate=0, flip_horizontal=False, flip_vertical=False, resize_scale=1.0, translate_x=0, translate_y=0):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)
    fps = cap.get(cv2.CAP_PROP_FPS) * speed

    base_name = os.path.basename(video_path).split('.')[0]
    save_path = get_next_filename(save_dir, base_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    print(f"비디오 증강 및 저장 중: {save_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        # Rotate
        if rotate != 0:
            center = (frame_width // 2, frame_height // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
            frame = cv2.warpAffine(frame, rot_matrix, (frame_width, frame_height))
        
        # Flip
        if flip_horizontal:
            frame = cv2.flip(frame, 1)
        if flip_vertical:
            frame = cv2.flip(frame, 0)
        
        # Translate
        if translate_x != 0 or translate_y != 0:
            trans_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
            frame = cv2.warpAffine(frame, trans_matrix, (frame_width, frame_height))

        out.write(frame)
        
        cv2.imshow('Augmented Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"비디오 증강 완료. 저장 경로: {save_path}")

def process_all_videos(root_dir, save_root_dir, speed=1.0, rotate=0, flip_horizontal=False, flip_vertical=False, resize_scale=1.0, translate_x=0, translate_y=0):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, root_dir)
                save_dir = os.path.join(save_root_dir, relative_path)
                augment_video(video_path, save_dir, speed, rotate, flip_horizontal, flip_vertical, resize_scale, translate_x, translate_y)

# 파라미터 설정
root_video_dir = './Video'
save_root_dir = './Augmented_Video'
speed = 1.0  # 속도 배율
rotate = 15  # 회전 각도
flip_horizontal = False  # 가로 뒤집기
flip_vertical = False  # 세로 뒤집기
resize_scale = 1.0  # 크기 조정 배율
translate_x = 0  # X축 이동
translate_y = 0  # Y축 이동

# 실행
process_all_videos(
    root_video_dir, 
    save_root_dir, 
    speed, 
    rotate, 
    flip_horizontal, 
    flip_vertical, 
    resize_scale, 
    translate_x, 
    translate_y
)
