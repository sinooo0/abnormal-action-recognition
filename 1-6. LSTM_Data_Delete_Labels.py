import os

def clean_labels_with_deleted_images(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if 'images' in dirs and 'labels' in dirs:
            image_dir = os.path.join(root, 'images')
            label_dir = os.path.join(root, 'labels')

            image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}

            # labels 하위 모든 폴더 탐색
            for label_root, _, label_files in os.walk(label_dir):
                label_files_set = {os.path.splitext(f)[0] for f in label_files if f.endswith('.csv')}

                # 이미지와 매칭되지 않는 라벨 파일 삭제
                unmatched_labels = label_files_set - image_files

                for label in label_files_set:
                    if label in unmatched_labels:
                        label_path = os.path.join(label_root, f"{label}.csv")
                        os.remove(label_path)
                        print(f"삭제된 라벨 파일: {label_path}")

# 실행
base_dir = './Data/LSTM_Capture'
clean_labels_with_deleted_images(base_dir)