# 👁️ VisionTrack — Object Detection & Tracking

**CodeAlpha AI Internship — Task 3 (Bonus)**

Real-time object detection and tracking using **YOLOv8** and **OpenCV**. Detects 80+ object classes (people, vehicles, animals, etc.) from your webcam or a video file, with optional tracking that assigns unique IDs to each object across frames.

---

## ✨ Features

- **Real-time detection** using YOLOv8 — state-of-the-art speed and accuracy
- **80+ object classes** — people, cars, dogs, books, phones, and more
- **Webcam or video file** input — switch with `--source`
- **Live tracking mode** (`--track`) — assigns persistent IDs to detected objects across frames
- **Interactive controls** — adjust confidence threshold live with `+` / `-` keys
- **Screenshot capture** — press `s` to save any frame

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core language |
| OpenCV | Video capture & frame rendering |
| Ultralytics YOLOv8 | Pre-trained object detection model |
| BoT-SORT (via YOLO) | Object tracking algorithm |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.8 or higher
- Webcam (optional — can use a video file instead)
- Git installed

### Step-by-step

```bash
# 1. Navigate to the project folder
cd CodeAlpha/Task3_ObjectDetectionTracking

# 2. (Recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run with webcam (default)
python app.py

# 5. Run with a video file
python app.py --source path/to/video.mp4

# 6. Run with tracking enabled
python app.py --track

# 7. Use a different YOLO model (nano is fastest, x-large is most accurate)
python app.py --model yolov8s.pt  # small
python app.py --model yolov8m.pt  # medium
python app.py --model yolov8x.pt  # x-large (needs GPU)
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save current frame as screenshot |
| `+` / `=` | Increase confidence threshold |
| `-` / `_` | Decrease confidence threshold |

---

## 📁 Project Structure

```
Task3_ObjectDetectionTracking/
├── app.py           # Main detection/tracking script
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## ⚠️ Notes

- First run downloads the YOLO model automatically (~6 MB for yolov8n.pt)
- For best performance on CPU, use `yolov8n.pt` (nano model). For accuracy, use `yolov8x.pt` (requires GPU)
- Tracking mode (`--track`) uses BoT-SORT internally via Ultralytics
- No GPU required for yolov8n / yolov8s — runs well on modern CPUs at 15–30 FPS

---

## 📄 License

This project is submitted as part of the **CodeAlpha AI Internship**.
