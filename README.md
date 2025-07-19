# abnormal-action-recognition
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-Compatible-green.svg)

## 📌 개요
- 실시간 웹캠 기반 행동 감지
- 다중 객체 추적 및 분석
- 4가지 행동 패턴 분류 (정상/의심/위험/흉기)
- FastAPI 기반 웹 인터페이스
- 데이터 증강 및 전처리 도구

## 🔧 환경 설정

### 🖥️ 사전 요구사항
- Python 3.9
- CUDA 지원 GPU (권장)
- Webcam (실시간 테스트용)

### 🛠️ Conda 가상환경 설정

#### 1. 가상환경 생성

```bash
conda create -n action_recognition python=3.9 -y
```

#### 2. 가상환경 실행

```bash
conda activate action_recognition
```

#### 3. 의존성 설치

`requirements.txt` 파일을 통해 필요한 라이브러리를 설치

```bash
pip install -r requirements.txt
```

## 📊 1. 데이터 생성

### 1-1. `LSTM_Data_Live.py`

- 웹캠에서 프레임을 캡처한 후 YOLO-pose로 키포인트를 추출해 CSV 파일로 저장
- py파일 내에서 저장할 시퀀스 길이를 지정함
- CSV 파일은 `LSTM_Live` 폴더 내 각 클래스 번호 폴더에 ID별로 저장됨

### 1-2. `Video_Record.py`

- 실시간 영상 녹화 후 저장
- 카메라 번호, 저장경로, 녹화시간(초) 설정 가능

### 1-3. `Video_Augmentation.py`

- 동영상 증강 후 Augmented_Video 폴더에 저장
- speed, rotate, flip, scale, translate 조정 가능

### 1-4. `Rename.py`

- 폴더 경로 안의 파일들 이름 통일
- {파일명}_{번호} 형식으로 오름차순 저장
- 중복되는 파일명 존재 시, 그 다음 번호부터 시작

### 1-5. `LSTM_Data_Capture.py`

- 동영상에서 프레임을 캡처한 후 YOLO-pose로 키포인트를 추출해 이미지와 각 ID의 CSV 파일로 저장
- 동영상이 들어있는 폴더 경로 입력 시, 동영상별 폴더를 생성해 이미지와 각 ID의 CSV 파일을 프레임별로 저장함

### 1-6. `LSTM_Data_Delete_Labels.py`

- 이미지와 같은 프레임의 CSV 파일만 남기고 나머지 CSV 파일 삭제
- `1-4`에서 필요한 이미지만 분류한 후 실행
- 시퀀스 단위로 이미지를 분류해야 학습할 때 문제가 발생하지 않음

## 🤖 2. 모델 학습

### 2-1. `LSTM_Train.ipynb`

- 로컬 환경에서 실행 가능
- `LSTM_Data` 폴더 내 모든 CSV 파일 병합
- 결합된 데이터를 LSTM 학습용 3차원 형식으로 변환: `(n, 시퀀스, 키포인트 수 + 클래스)`
- 모델 정의 → 학습 → 저장

### 2-2. `YOLO_Train.ipynb`

- Colab GPU 환경에서 실행
- 데이터셋 로드 및 정제
- YOLO 모델 학습 및 저장

### 🌟 LSTM_Train

- Colab GPU 환경에서 실행
- Video 데이터 셋 불러와서 전처리 및 학습 수행
- LSTM 학습을 위한 1~2 과정을 한 번에 수행 가능

## 🌐 3. 배포

### 3-1. `Fast_API.py`

- LSTM과 YOLO 모델을 결합하여 한 화면에 두개의 예측 동시 수행
- YOLO-pose로 다중 객체 탐지 후 ID 및 키포인트 수집
- 슬라이딩 윈도우 방식으로 시퀀스 구성
- 시퀀스 완성 시 ID별로 LSTM을 통해 행동 예측
- 이후 프레임 변화마다 실시간 예측을 반복 수행
- YOLO-pose와 LSTM을 멀티 스레딩으로 분리해 빠른 예측 가능
- /webcam에서 Webcam 프레임 전달받음
- Danger/Weapon 탐지 시 이미지 전송
- 원본 이미지, 탐지 이미지 각각 전송 가능
- /predict에서 실시간 탐지 영상 확인 가능

### 3-2. `Webcam.py`

- Fast_API 서버로 카메라 프레임 전송

### 🌟 API_Server

- Goorm IDE에서 실행
- setup.txt 파일 참고하여 컨테이너 초기 설정
- GPU 환경에서 API 서버 호스팅
- 실시간 데이터 전송 및 예측 수행

## 🏷️ 클래스 분류

| 클래스 번호 | 행동 분류  | 세부 동작     |
| ------ | ------ | -------------------- |
| 0      | normal  | 서있기, 걷기, 뛰기, 앉기 |
| 1      | doubt  | 가방 열기, 두리번거리기 |
| 2      | danger  | 잽, 훅, 발차기, 어깨잡기, 뺏기|
|        | weapon | 흉기                   |

## ⚙️ 시스템 요구사항

### 권장 요구사항
- CPU: Intel i7 또는 유사사양
- GPU: NVIDIA T4 GPU
- RAM: 16GB
- 저장소: 20GB

## ⚠️ 알려진 문제 및 제한 사항

#### 1. TensorFlow/Keras 버전 호환성
- TF 2.15.0 이하: Keras 2.x 사용
- TF 2.16.0 이상: Keras 3.x 필요
- 버전 간 모델 호환 불가

#### 2. 데이터 처리 주의사항
- 시퀀스 단위로 데이터 정리 필요
- 프레임 동기화 확인 필요

## 💡 모델 학습 이슈 사항
 
### 1. 범죄 데이터 부족
- **문제**: 범죄 데이터를 충분히 확보하지 못해, 학습에 필요한 데이터가 부족함
- **해결**: 데이터 생성 및 증강을 통해 범죄 관련 데이터셋을 확장하여 해결
 
### 2. 구석으로 가면 Danger로 인식
- **문제**: 구석에 있는 인물들이 Danger로 잘못 분류되는 현상이 발생
- **해결**: 일정 키포인트 이상만 탐지하도록 모델을 수정하여 정확도를 개선
 
### 3. 키포인트를 절대 좌표 기준으로 추출하여 위치기반으로 판단
- **문제**: 키포인트를 절대 좌표 기준으로 추출하여, 행동이 아닌 위치 기반으로 판단되는 오류 발생
- **해결**: 바운딩 박스를 기준으로 키포인트를 정규화하여 문제를 해결
 
### 4. 다중 모델 구현 및 실시간 처리 문제
- **문제**: GPU 서버에서 다중 모델을 동시에 실행할 때, 초당 3프레임으로 행동 인식이 이루어지는 문제가 발생
- **해결**: 시퀀스 프레임 수를 증가시켜 초당 10프레임으로 행동을 판단하도록 설정
 
### 5. 학습한 인물이 아닌 다른 인물 탐지 부정확
- **문제**: 학습한 인물이 아닌 다른 인물의 행동 탐지 정확도가 떨어짐
- **해결**: 어깨 및 골반을 기준으로 키포인트를 정규화하여 신체 비율에 따른 오차를 최소화
