import pandas as pd
import matplotlib.pyplot as plt
import os

# 관절 연결 구조 (YOLO-Pose 기준)
connections = [
    (0, 1), (1, 3), (0, 2), (2, 4),      # 얼굴 (코-눈-귀)
    (0, 5), (5, 7), (7, 9),              # 왼쪽 팔
    (0, 6), (6, 8), (8, 10),            # 오른쪽 팔
    (5, 6),                              # 왼쪽 어깨-오른쪽 어깨 연결
    (11, 12),                            # 왼쪽 엉덩이-오른쪽 엉덩이 연결
    (5, 11), (6, 12),                   # 어깨-엉덩이 연결
    (11, 13), (13, 15),                 # 왼쪽 다리
    (12, 14), (14, 16)                  # 오른쪽 다리
]

# 키포인트 이름
keypoint_labels = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

# 부위별 색상 매핑
colors = {
    'head': '#4CAF50',
    'arm': '#2196F3',
    'leg': '#FF9800'
}

def get_color(index):
    if index in [0, 1, 2, 3, 4]:  # 머리
        return colors['head']
    elif index in [5, 6, 7, 8, 9, 10]:  # 팔
        return colors['arm']
    else:  # 다리
        return colors['leg']

def visualize_keypoints(input_folder, output_root):
    # 출력 폴더 생성
    output_folder = os.path.join(output_root, os.path.basename(input_folder))
    os.makedirs(output_folder, exist_ok=True)

    # 폴더 내 모든 CSV 파일 탐색 및 처리
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                keypoints_df = pd.read_csv(csv_path)

                # 원본 CSV 파일과 동일한 경로에 이미지 저장
                save_dir = os.path.join(output_folder, os.path.relpath(root, input_folder))
                os.makedirs(save_dir, exist_ok=True)
                
                plt.figure(figsize=(6, 10))  # 비율 조정
                for idx, row in keypoints_df.iterrows():
                    keypoints = [(row[f'kp{i}_x'], row[f'kp{i}_y']) for i in range(17)]

                    for i, (x, y) in enumerate(keypoints):
                        if x != 0 and y != 0:
                            plt.scatter(x, y, s=50, color=get_color(i))
                            plt.text(x, y, keypoint_labels[i], fontsize=8, color='black')
                    for start, end in connections:
                        x1, y1 = keypoints[start]
                        x2, y2 = keypoints[end]
                        if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                            plt.plot([x1, x2], [y1, y2], color=get_color(start), linewidth=2)
                    plt.gca().invert_yaxis()
                    plt.axis('off')
                    plt.gca().set_aspect('equal', adjustable='box')  # 비율 고정
                    plt.subplots_adjust(top=0.9)  # 제목 공간 확보
                    plt.title(f'Keypoints Visualization - {file}')

                    # CSV 파일 이름으로 이미지 저장
                    save_path = os.path.join(save_dir, f'{os.path.splitext(file)[0]}.png')
                    plt.savefig(save_path, dpi=300)
                    plt.close()
    print('All CSV keypoint images have been saved.')

if __name__ == "__main__":
    input_folder = './Data/LSTM_Augmentation'  # CSV 파일들이 있는 폴더 경로
    output_root = './Data/Keypoint_Images'  # 이미지 저장 루트 경로
    visualize_keypoints(input_folder, output_root)
