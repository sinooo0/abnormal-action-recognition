import cv2
import requests
import threading
import time
from datetime import datetime

# 서버 URL 및 세션 생성
server_url = "https://crime-detect.run.goorm.site/webcam"
session = requests.Session()
last_timestamp = time.time()
error_occurred = False

def send_frame(frame):
    global last_timestamp, error_occurred
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ret:
        return
    files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
    try:
        response = session.post(server_url, files=files, timeout=0.5)
        if response.status_code == 200:
            now = time.time()
            if now - last_timestamp >= 5:
                print(f"프레임 전송 성공: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                last_timestamp = now
            error_occurred = False
    except Exception as e:
        if not error_occurred:
            print("Error sending frame:", e)
            error_occurred = True

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 캡처 실패")
                break

            # 프레임 전송을 별도 스레드에서 수행
            threading.Thread(target=send_frame, args=(frame,), daemon=True).start()

            key = cv2.waitKey(1)
            if key == 27:  # ESC 눌러서 종료
                break
    finally:
        cap.release()

if __name__ == "__main__":
    main()
