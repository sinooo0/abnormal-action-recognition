# YOLO_LSTM

## 개요
1. 웹캠이나 동영상에서 캡처한 프레임을 YOLO-pose로 처리해 키포인트 데이터를 수집
2. LSTM을 활용해 키포인트 데이터 시퀀스를 기반으로 행동을 예측

---

## 0. 환경 설정

### 0-1. Conda 가상환경 생성

```bash
conda create -n action_recognition python=3.12 -y
```

### 0-2. Conda 가상환경 실행

```bash
conda activate action_recognition
```

### 0-3. 의존성 설치

`requirements.txt` 파일을 통해 필요한 라이브러리를 설치합니다:

```bash
pip install -r requirements.txt
```

---

## 1. 데이터 생성

### 1-1. `LSTM_Data_Live.py`

- 웹캠에서 프레임을 캡처한 후 YOLO-pose로 키포인트를 추출해 CSV 파일로 저장
- py파일 내에서 저장할 시퀀스 길이를 지정함
- CSV 파일은 `LSTM_Live` 폴더 내 각 클래스 번호 폴더에 ID별로 저장됨

### 1-2. `LSTM_Data_Capture.py`

- 동영상에서 프레임을 캡처한 후 YOLO-pose로 키포인트를 추출해 이미지와 각 ID의 CSV 파일로 저장
- 동영상이 들어있는 폴더 경로 입력 시, 동영상별 폴더를 생성해 이미지와 각 ID의 CSV 파일을 프레임별로 저장함

### 1-3. `LSTM_Data_Delete_Labels.py`

- 이미지와 같은 프레임의 CSV 파일만 남기고 나머지 CSV 파일 삭제
- `1-2`에서 필요한 이미지만 분류한 후 실행
- 시퀀스 단위로 이미지를 분류해야 학습할 때 문제가 발생하지 않음

### 1-4. `Video_Record.py`

- 실시간 영상 녹화 후 저장
- 카메라 번호, 저장경로, 녹화시간(초) 설정 가능

### 1-5. `Video_Augmentation.py`

- 동영상 증강 후 Augmented_Video 폴더에 저장
- speed, rotate, flip, scale, translate 조정 가능

### 1-6. `Keypoint_Visualize.py`

- CSV 파일을 읽어 키포인트 시각화
- 여러 행의 CSV 파일을 읽으면 코드 실행되는 동안 이미지가  순차적으로 변하며, 마지막 이미지가 저장됨

### 1-7. `Rename.py`

- 폴더 경로 안의 파일들 이름 통일
- {파일명}_{번호} 형식으로 오름차순 저장
- 중복되는 파일명 존재 시, 그 다음 번호부터 시작
---

## 2. 모델 학습

### 2-1. `LSTM_Train.ipynb`

- 로컬 환경에서 실행 가능
- `LSTM_Data` 폴더 내 모든 CSV 파일 병합
- 결합된 데이터를 LSTM 학습용 3차원 형식으로 변환: `(n, 시퀀스, 키포인트 수 + 클래스)`
- 모델 정의 → 학습 → 저장

### 2-2. `YOLO_Train.ipynb`

- Colab GPU 환경에서 실행
- 데이터셋 로드 및 정제
- YOLO 모델 학습 및 저장

---

## 3. 실행

### 3-1. `LSTM_Test.py`

- YOLO-pose로 다중 객체 탐지 후 ID 및 키포인트 수집
- 슬라이딩 윈도우 방식으로 시퀀스 구성
- 시퀀스 완성 시 ID별로 LSTM을 통해 행동 예측
- 이후 프레임 변화마다 실시간 예측을 반복 수행
- YOLO-pose와 LSTM을 멀티 스레딩으로 분리해 빠른 예측 가능

### 3-2. `YOLO_Test.py`

- 실시간으로 다중 YOLO 모델 예측 가능

---

## 클래스 분류

| 클래스 번호 | 행동 분류  | 세부 동작                |
| ------ | ------ | -------------------- |
| 0      | normal | 가만히 서있기, 걷기, 뛰기, 앉기  |
| 1      | fight  | 잽, 훅, 발차기, 파운딩, 어깨잡기 |
| 2      | theft  | 가방 뺏기                |
| 3      | unbag  | 가방 열기                |
| 4      | scan   | 두리번거리기               |
| 5      | weapon rampage   | 흉기난동            |

---

## 사용 환경

- YOLO-pose 모델 사용
- LSTM 모델 사용
- Colab GPU 환경 (YOLO 학습)
- 로컬 환경 (LSTM 학습 및 테스트)

---

## 실행 방법

1. Conda 가상환경 생성 및 라이브러리 설치
2. 데이터 생성 스크립트를 실행해 학습 데이터 준비
3. LSTM 및 YOLO 모델 학습
4. 실시간 예측 실행
