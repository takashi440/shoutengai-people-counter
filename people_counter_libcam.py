# === 更新履歴 ===
# 2025-06-20: IMX219（960x720）対応、frame_size修正済
# 2025-06-20: YOLO推論の負荷軽減対応（15FPS・インターバル・軽量モデル）
# 2025-06-25: person判定ロジックを厳密化（YOLOでpersonと明示されたもののみを「人」としてカウント）

import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import sys
import logging
import psutil
import os
from flask import Flask, Response
import threading

app = Flask(__name__)
latest_frame = None

# === ログ設定 ===
logging.basicConfig(
    filename='debug.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# === DeepSORTライブラリの読み込み ===
sys.path.append('./deepsort_public')
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection

# === Google Sheets API設定 ===
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("path/to/credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("YourSpreadsheetName").worksheet("YourSheetTab")

# === DeepSORT 初期化 ===
metric = NearestNeighborDistanceMetric("cosine", 0.4, 100)
tracker = Tracker(metric)
model = YOLO("yolov8n.pt")
model_classes = model.names

# === カウント設定 ===
non_person_classes = {"car", "bus", "truck", "bicycle", "motorcycle", "bench", "chair", "suitcase"}
counted_ids = set()
track_history = {}
track_categories = {}
pass_count = {
    "person_LR": 0, "person_RL": 0,
    "bicycle_LR": 0, "bicycle_RL": 0,
    "car_LR": 0, "car_RL": 0,
}

# === カメラ設定 ===
width, height = 640, 480
frame_size = int(width * height * 1.5)
frame_timeout = 10
inference_interval = 0.3
last_inference_time = 0
start_time = time.time()

# === 通過方向判定 ===
def get_horizontal_direction(track_id, threshold=70):
    history = track_history.get(track_id, [])
    if len(history) < 2:
        return None
    delta = history[-1] - history[0]
    logging.info(f"↔ track_id={track_id}, delta={delta}")
    if delta > threshold:
        return "LR"
    elif delta < -threshold:
        return "RL"
    return None

# === システム状態ログ ===
def log_system_status():
    temp = subprocess.getoutput("vcgencmd measure_temp")
    mem = psutil.virtual_memory()
    logging.info(f"[SYSTEM] Temp: {temp}, MemUsage: {mem.percent}%, Free: {mem.available // 1024 // 1024}MB")

# === モニタリング映像出力 ===
def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === カラーバランス補正（赤外線用：通常カメラでは軽微な影響） ===
def correct_infrared_color(frame):
    b, g, r = cv2.split(frame)
    r = cv2.subtract(r, 70)
    g = cv2.add(g, 10)
    return cv2.merge((b, g, r))

# === カメラ実行ロジック ===
def run_camera():
    global latest_frame, last_inference_time, start_time
    cmd = [
        "libcamera-vid", "--width", str(width), "--height", str(height), "--framerate", "15",
        "--codec", "yuv420", "-o", "-", "--timeout", "0"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    print("📷 通過カウント＋モニタリングを開始しました")

    while True:
        try:
            start_read = time.time()
            raw_data = process.stdout.read(frame_size)
            if time.time() - start_read > frame_timeout or not raw_data:
                break

            yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape((int(height * 1.5), width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            bgr = correct_infrared_color(bgr)

            now = time.time()
            if now - last_inference_time < inference_interval:
                latest_frame = bgr.copy()
                continue
            last_inference_time = now

            print("🔍 YOLO推論中...")
            results = model(bgr, verbose=False, conf=0.2)
            print("✅ YOLO推論完了")

            detections = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    feature = np.array([x1, y1, x2, y2, conf] * 26, dtype=np.float32)[:128]
                    det = Detection(bbox, conf, cls, feature)
                    det.cls_id = cls
                    detections.append(det)

            tracker.predict()
            tracker.update(detections)

            active_tracks = 0
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                active_tracks += 1
                track_id = track.track_id
                x, y, w, h = track.to_tlwh()
                cx = int(x + w / 2)

                if track_id not in track_history:
                    track_history[track_id] = []
                    for det in detections:
                        distance = np.linalg.norm(track.mean[:2] - np.array([det.tlwh[0], det.tlwh[1]]))
                        if distance < 40:
                            cls_name = model_classes.get(det.cls_id, "unknown")
                            if cls_name == "person":
                                track_categories[track_id] = "person"
                            elif cls_name in non_person_classes:
                                track_categories[track_id] = cls_name
                            else:
                                track_categories[track_id] = "unknown"
                            break
                    else:
                        track_categories[track_id] = "unknown"

                track_history[track_id].append(cx)
                direction = get_horizontal_direction(track_id)
                if direction and track_id not in counted_ids:
                    obj_type = track_categories.get(track_id, "unknown")
                    key = f"{obj_type}_{direction}"
                    if key in pass_count:
                        pass_count[key] += 1
                        counted_ids.add(track_id)
                        logging.info(f"ID:{track_id} ({obj_type}) → {direction} → {key}: {pass_count[key]}")

                label = f"{track_categories.get(track_id, 'ID')}:{track_id}"
                cv2.rectangle(bgr, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
                cv2.putText(bgr, label, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            print(f"👣 アクティブトラック数: {active_tracks}")
            latest_frame = bgr.copy()

            if now - start_time >= 600:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                try:
                    sheet.append_row([
                        timestamp,
                        pass_count["person_LR"], pass_count["person_RL"],
                        pass_count["bicycle_LR"], pass_count["bicycle_RL"],
                        pass_count["car_LR"], pass_count["car_RL"]
                    ])
                    logging.info(f"📝 記録成功: {timestamp} → {pass_count}")
                except Exception as e:
                    logging.error(f"スプレッドシート書き込み失敗: {e}")
                log_system_status()
                start_time = now
                counted_ids.clear()
                track_history.clear()
                track_categories.clear()
                pass_count.update({k: 0 for k in pass_count})
        except Exception as e:
            logging.exception("❌ 予期せぬエラーが発生しました")
            break

if __name__ == '__main__':
    threading.Thread(target=run_camera, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)
