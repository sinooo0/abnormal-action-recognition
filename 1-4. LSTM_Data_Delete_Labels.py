import os

def clean_labels_with_deleted_images(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if 'images' in dirs and 'labels' in dirs:
            image_dir = os.path.join(root, 'images')
            label_dir = os.path.join(root, 'labels')
            
            image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
            label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.csv')}
            
            # 이미지 파일 이름과 매칭 (ID 제외)
            image_base_files = {"_".join(f.split("_")[:2]) for f in image_files}
            label_base_files = {"_".join(f.split("_")[:2]) for f in label_files}
            
            # 이미지와 매칭되지 않는 라벨 파일 삭제
            unmatched_labels = label_base_files - image_base_files
            
            for label in label_files:
                label_base = "_".join(label.split("_")[:2])
                if label_base in unmatched_labels:
                    label_path = os.path.join(label_dir, f"{label}.csv")
                    os.remove(label_path)
                    print(f"삭제된 라벨 파일: {label_path}")

# 실행
base_dir = './Capture'
clean_labels_with_deleted_images(base_dir)
