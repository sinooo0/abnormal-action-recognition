import numpy as np
import pandas as pd
import random
import os

# CSV 파일에서 키포인트 데이터를 불러오는 함수
def load_keypoint_data(file_path):
    return pd.read_csv(file_path)

# 키포인트와 action_class 분리 함수
def split_keypoints_and_label(data):
    keypoints = data[:, :-1]  # 마지막 열 제외 (action_class)
    labels = data[:, -1].astype(int)  # 마지막 열 (action_class), 정수형으로 변환
    return keypoints, labels

# 회전 증강 함수 (각도: 도 단위)
def rotate_sequence(sequence, angle):
    rad = np.deg2rad(angle)
    cos_angle, sin_angle = np.cos(rad), np.sin(rad)
    rotated_sequence = []
    for frame in sequence:
        keypoints = frame.reshape(-1, 2)
        rotated = keypoints @ np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_sequence.append(rotated.flatten())
    return np.array(rotated_sequence)

# 스케일링 증강 함수 (배율)
def scale_sequence(sequence, scale_factor):
    scaled_sequence = []
    for frame in sequence:
        keypoints = frame.reshape(-1, 2)
        scaled = keypoints * scale_factor
        scaled_sequence.append(scaled.flatten())
    return np.array(scaled_sequence)

# 이동 증강 함수 (픽셀 단위)
def translate_sequence(sequence, shift_x, shift_y):
    translated_sequence = []
    for frame in sequence:
        keypoints = frame.reshape(-1, 2)
        translated = keypoints + [shift_x, shift_y]
        translated_sequence.append(translated.flatten())
    return np.array(translated_sequence)

# 가우시안 노이즈 추가 함수 (노이즈 강도)
def add_noise_sequence(sequence, noise_level=2):
    noisy_sequence = []
    for frame in sequence:
        noise = np.random.normal(0, noise_level, frame.shape)
        noisy = frame + noise
        noisy_sequence.append(noisy)
    return np.array(noisy_sequence)

# X축 또는 Y축 기준 반전 함수 (둘 다 가능)
def flip_sequence(sequence, x_flip=False, y_flip=False):
    flipped_sequence = []
    for frame in sequence:
        keypoints = frame.reshape(-1, 2)
        if x_flip:
            keypoints[:, 1] = -keypoints[:, 1]  # X축 기준 반전
        if y_flip:
            keypoints[:, 0] = -keypoints[:, 0]  # Y축 기준 반전
        flipped_sequence.append(keypoints.flatten())
    return np.array(flipped_sequence)

# 전체 증강 함수 (회전, 스케일링, 이동, 노이즈, 반전)
def augment_sequence(sequence, angle_range=(-10, 10), scale_range=(0.9, 1.1), shift_range=(-10, 10), noise_level=2, x_flip=False, y_flip=False):
    angle = random.uniform(*angle_range)  # 회전 범위 (도)
    scale_factor = random.uniform(*scale_range)  # 스케일 범위 (배율)
    shift_x, shift_y = random.uniform(*shift_range), random.uniform(*shift_range)  # 이동 범위 (픽셀)

    sequence = rotate_sequence(sequence, angle)
    sequence = scale_sequence(sequence, scale_factor)
    sequence = translate_sequence(sequence, shift_x, shift_y)
    sequence = add_noise_sequence(sequence, noise_level)
    sequence = flip_sequence(sequence, x_flip, y_flip)
    return sequence

# 증강 데이터를 저장하는 함수
def save_augmented_data(augmented_data, labels, original_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(original_file))[0]
    save_path = os.path.join(save_dir, f'{file_name}_Aug.csv')
    combined_data = np.hstack((augmented_data, labels.reshape(-1, 1)))
    columns = [f'kp{i}_{axis}' for i in range(17) for axis in ['x', 'y']] + ['action_class']
    df = pd.DataFrame(combined_data, columns=columns)
    df['action_class'] = df['action_class'].astype(int)
    df.to_csv(save_path, index=False)

# 전체 데이터를 증강하고 저장하는 함수
def process_and_augment(file_path, save_dir, angle_range=(-10, 10), scale_range=(0.9, 1.1), shift_range=(-10, 10), noise_level=2, x_flip=False, y_flip=False):
    data = load_keypoint_data(file_path).values
    if len(data) == 0:
        print(f"{file_path} 파일은 데이터가 없습니다.")
        return
    keypoints, labels = split_keypoints_and_label(data)
    augmented_keypoints = augment_sequence(keypoints, angle_range, scale_range, shift_range, noise_level, x_flip, y_flip)
    save_augmented_data(augmented_keypoints, labels, file_path, save_dir)

# 모든 CSV 파일을 탐색하고 증강하는 함수
def augment_all_csv_in_directory(root_dir, save_dir, angle_range=(-10, 10), scale_range=(0.9, 1.1), shift_range=(-10, 10), noise_level=2, x_flip=False, y_flip=False):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                process_and_augment(file_path, save_dir, angle_range, scale_range, shift_range, noise_level, x_flip, y_flip)

# 실행
file_path = './Data/LSTM_Capture'
save_dir = './Data/LSTM_Augmentation'
angle_range = (-15, 15)  # 회전 범위 (각도)
scale_range = (0.8, 1.2)  # 스케일 범위 (배율)
shift_range = (-15, 15)  # 이동 범위 (픽셀)
noise_level = 3  # 노이즈 강도
x_flip = False  # X축 기준 반전 여부
y_flip = False  # Y축 기준 반전 여부

augment_all_csv_in_directory(file_path, save_dir, angle_range, scale_range, shift_range, noise_level, x_flip, y_flip)
