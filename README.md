# 👣 People Counter System with YOLOv8 + DeepSORT + Raspberry Pi

Raspberry Pi + 通常カメラ（例：IMX219）を使い、YOLOv8とDeepSORTによって「人・自転車・車などの通過人数」を左右の移動方向ごとにカウントし、10分ごとにGoogle Sheetsへ自動記録します。リアルタイムモニタリング機能も搭載。

---

## 📷 システム概要

- **プラットフォーム**: Raspberry Pi (推奨: Pi 4 or 5)
- **カメラ**: IMX219 または V2 Camera Module
- **解像度**: 640x480
- **物体検出**: YOLOv8n（軽量モデル）
- **トラッキング**: DeepSORT
- **出力先**: Google Sheets
- **サーバー機能**: Flask（ポート8080でJPEGストリーム配信）

---

## 🔁 動作フロー

1. **カメラ起動**: `libcamera-vid` を使用し、YUV形式で映像ストリーミング
2. **画像処理**: YUV → BGR変換＋赤外補正（※通常カメラ時は補正軽微）
3. **物体検出**: YOLOv8nにより、人・自転車・車などを検出
4. **トラッキング**: DeepSORTでID追跡
5. **通過方向判定**: x方向±70pxの移動でLR/RL判定
6. **集計と記録**: 10分ごとにGoogle Sheetsへ自動追記
7. **モニタリング**: `http://<RaspberryPiのIP>:8080` にて映像確認

---

## 🔧 パラメータ設定

| パラメータ名 | 値 | 説明 |
|--------------|----|------|
| `width`, `height` | 640x480 | 画角と処理効率のバランス |
| `inference_interval` | 0.3秒 | 推論頻度（約3FPS） |
| `conf` | 0.2 | YOLOの検出信頼度 |
| `threshold` | 70 px | 通過判定の移動量 |
| `recording_interval` | 600秒 | Google Sheets書き込み周期 |
| `frame_timeout` | 10秒 | カメラフリーズ検出用 |

---

## 📦 ディレクトリ構成（例）

