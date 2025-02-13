import cv2
import os

def get_next_filename(directory, extension='.mp4'):
    os.makedirs(directory, exist_ok=True)
    base_name = os.path.basename(directory).split('. ', 1)[-1]  # 폴더명에서 클래스 번호 제외
    existing_files = [f for f in os.listdir(directory) if f.startswith(base_name) and f.endswith(extension)]
    numbers = [int(f[len(base_name):-len(extension)]) for f in existing_files if f[len(base_name):-len(extension)].isdigit()]
    next_number = max(numbers, default=0) + 1
    return os.path.join(directory, f"{base_name}{next_number}{extension}")

def record_video(camera_index, save_dir, duration):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"웹캠 {camera_index}을(를) 열 수 없습니다.")
        return

    # 비디오 프레임 설정
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 30.0

    # 저장 파일명 생성
    save_path = get_next_filename(save_dir)

    # 비디오 코덱 및 VideoWriter 객체 생성 (MP4 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"녹화 시작: {duration}초 동안 녹화합니다.")

    frame_count = 0
    max_frames = int(fps * duration)
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("녹화를 중지합니다.")
                break
            
            frame_count += 1
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"녹화 완료. 파일이 저장되었습니다: {save_path}")

# 실행
camera_index = 1 # 카메라 번호 설정
save_path = './Video/2. danger' # 저장 경로 설정
duration = 5 # 녹화 시간(초) 설정
record_video(camera_index, save_path, duration)
