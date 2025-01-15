import os

def clean_labels_with_deleted_images(base_dir):
    # 하위 폴더 탐색
    for root, dirs, files in os.walk(base_dir):
        if 'images' in dirs and 'labels' in dirs:
            image_dir = os.path.join(root, 'images')
            label_dir = os.path.join(root, 'labels')
            
            # 이미지 파일 목록 가져오기 (확장자 제외)
            image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
            
            # 라벨 파일 목록 가져오기 (확장자 제외)
            label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.csv')}
            
            # 이미지와 매칭되지 않는 라벨 파일 삭제
            unmatched_labels = label_files - image_files
            for label in unmatched_labels:
                label_path = os.path.join(label_dir, f"{label}.csv")
                os.remove(label_path)
                print(f"삭제된 라벨 파일: {label_path}")

# 실행
base_dir = './Capture'  # 최상위 폴더 경로
clean_labels_with_deleted_images(base_dir)
