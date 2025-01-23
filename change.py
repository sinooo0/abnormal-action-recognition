import os
import re
import pandas as pd

def update_action_class(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # LSTM_Capture 바로 아래 폴더 이름에서 클래스 번호 추출
                relative_path = os.path.relpath(file_path, base_path)
                top_folder = relative_path.split(os.sep)[0]  # 최상위 폴더 추출

                # 최상위 폴더 이름에서 숫자 추출
                match = re.match(r"(\d+)", top_folder)  # 폴더 이름의 시작 부분에 있는 숫자를 찾음
                if match:
                    class_number = int(match.group(1))
                else:
                    print(f"Skipping file: {file_path} (Invalid top folder name format: {top_folder})")
                    continue

                # CSV 파일 수정
                try:
                    df = pd.read_csv(file_path)
                    if 'action_class' in df.columns:
                        df['action_class'] = class_number
                        df.to_csv(file_path, index=False)
                        print(f"Updated: {file_path}")
                    else:
                        print(f"Skipping file: {file_path} ('action_class' column not found)")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

# 경로 지정 및 실행
base_path = "./Data/LSTM_Capture"
update_action_class(base_path)
