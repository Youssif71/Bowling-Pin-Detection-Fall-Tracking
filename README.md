# 🎳 Bowling Pin Detection & Fall Tracking

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8s-00FFFF?style=flat&logo=ultralytics&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-Private-red?style=flat)

> A computer vision pipeline for real-time bowling pin detection, persistent tracking, and automated fall detection using a custom-trained YOLOv8 model.

**[Features](#-features) • [Installation](#️-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Results](#-results)**

---

## 🎯 Overview

This pipeline fine-tunes a **YOLOv8s** model on a custom-annotated dataset of 323 bowling frames, then applies it to a full bowling video to:

- 🎳 **Detect** every visible pin in each frame
- 🔢 **Register** each pin on first appearance with a persistent ID
- 📉 **Track** pins across frames using centroid matching
- ✅ **Classify** each pin as standing or fallen via aspect-ratio analysis and fall-streak confirmation
- 🎬 **Render** an annotated output video with a final summary screen

---

## ✨ Features

### 🔬 Detection & Tracking
- Single-class YOLOv8s model fine-tuned specifically for bowling pins
- BoTSort tracker (`botsort.yaml`) for stable multi-object association
- Configurable confidence threshold and match radius

### 📊 Intelligent Fall Detection
- **Aspect-ratio analysis** — pins are tall when standing; wide/square when fallen
- **Fall-streak debouncing** — a pin must appear fallen for `FALL_CONFIRM = 4` consecutive frames before being confirmed
- **Spatial anchoring** — fallen pins are fixed at their last known position to survive tracking ID resets
- **Baseline calibration** — the model observes each pin for `BASELINE_FRAMES = 8` frames before judging its state
- **Dual match radius** — standing pins use a tight 120 px radius; fallen pins use a wider 400 px radius to handle rolling

### 🎬 Video Output
- Frame-by-frame bounding box overlays with pin ID labels
- Elapsed time and fallen pin count overlay
- Final summary screen (2-second hold) showing total fallen count and video duration

---

## 🛠️ Installation

**Requirements:** Python 3.8+, CUDA-capable GPU (recommended)

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/bowling-pin-detection.git
cd bowling-pin-detection
```

### Step 2 — Install dependencies
```bash
pip install ultralytics opencv-python scikit-learn
```

### Step 3 — Verify
```bash
python -c "from ultralytics import YOLO; print('✓ Ready')"
```

---

## 🚀 Usage

### Run the full notebook

Open `notebook/bowling_detection.ipynb` and run all cells in order. The notebook is self-contained and walks through every stage of the pipeline.

### Quick inference on a new video

```python
from ultralytics import YOLO
import cv2

# Load best checkpoint
model = YOLO("notebook/runs/detect/bowling_pin_only/weights/best.pt")

# Run on a video — saves annotated output automatically
model.predict(source="videos/input.MOV", save=True, conf=0.4)
```

### Run the tracking pipeline directly

```python
import cv2
from ultralytics import YOLO

model = YOLO("notebook/runs/detect/bowling_pin_only/weights/best.pt")

input_video  = "videos/input.MOV"
output_video = "videos/output_pins_only.mp4"

cap = cv2.VideoCapture(input_video)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out    = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# … tracking loop — see notebook Cell 15 for the full implementation
```

---

## ⚙️ Configuration

All key parameters live in **Cell 14** of the notebook:

| Parameter | Value | Description |
|---|---|---|
| `MATCH_RADIUS_STANDING` | `120` px | Max centroid distance to re-associate a standing pin |
| `MATCH_RADIUS_FALLEN` | `400` px | Wider radius for fallen/rolling pins |
| `FALL_CONFIRM` | `4` frames | Consecutive fallen frames before confirming a fall |
| `BASELINE_FRAMES` | `8` frames | Frames used to calibrate each pin's baseline W/H ratio |
| `INIT_WINDOW_SECS` | `6` s | Time window for initializing the pin registry |
| `PIN_CLASS` | `0` | YOLO class index for `bowling-pin` |

Training configuration (`args.yaml`):

| Parameter | Value |
|---|---|
| Base model | `yolov8s.pt` (COCO pretrained) |
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 4 |
| Optimizer | Auto (AdamW) |
| Patience | 15 epochs |
| AMP | Enabled |
| Tracker | BoTSort |

---

## 📁 Project Structure

```
Computer_Vision_Main_Project/
│
├── dataset/
│   ├── Find pin.yolov8/              # Raw Roboflow export (323 images, YOLOv8 format)
│   │   ├── README.roboflow.txt
│   │   ├── data.yaml
│   │   └── train/
│   │       ├── images/
│   │       └── labels/
│   │
│   └── bowling_pins_dataset/         # Processed split dataset
│       ├── data.yaml
│       ├── train/                    # 259 images  (80%)
│       ├── valid/                    # 33 images   (10%)
│       └── test/                     # 34 images   (10%)
│
├── notebook/
│   ├── bowling_detection.ipynb       # Full pipeline notebook (16 cells)
│   ├── yolov8s.pt                    # Base weights
│   ├── yolo26n.pt                    # Reference weights
│   └── runs/
│       └── detect/
│           └── bowling_pin_only/
│               ├── args.yaml         # Training configuration
│               ├── results.csv       # Per-epoch metrics log
│               └── weights/
│                   ├── best.pt       # Best checkpoint (by mAP50)
│                   └── last.pt       # Final epoch checkpoint
│
└── videos/
    ├── input.MOV                     # Raw bowling footage
    └── output_pins_only.mp4          # Annotated output
```

### Dataset format (standard YOLO)

```
bowling_pins_dataset/
├── train/
│   ├── images/   ← .jpg frames
│   └── labels/   ← .txt bounding boxes (normalized xywh)
├── valid/
└── test/
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Roboflow](https://roboflow.com) — custom annotation |
| Total images | 323 frames from real bowling footage |
| Class | `bowling-pin` (single class) |
| Format | YOLOv8 normalized bounding boxes |
| Pre-processing | None |
| Augmentation | None |

---

## 📈 Results

Final validation metrics after 50 epochs:

| Metric | Value |
|---|---|
| **Precision** | **99.4%** |
| **Recall** | **98.0%** |
| **mAP@50** | **99.5%** |
| **mAP@50-95** | **92.1%** |

The model converged quickly — by epoch 2 it already achieved >99% precision and mAP@50 > 0.99. mAP@50-95 continued improving through all 50 epochs, reaching **0.921** at the final checkpoint.

Full per-epoch logs are in `notebook/runs/detect/bowling_pin_only/results.csv`.

---

## 🔧 How It Works

```
Input Video
    │
    ▼
Frame extraction  (OpenCV VideoCapture)
    │
    ▼
YOLOv8s + BoTSort  ──►  Bounding boxes + tracking IDs per frame
    │
    ▼
Pin registry initialization  (first INIT_WINDOW_SECS seconds)
    │  Each detected pin gets a unique persistent ID
    ▼
Per-frame matching
    │  New detections matched to registered pins by centroid distance
    │  Unmatched detections in the init window → new pin entries
    ▼
Fall detection
    │  Aspect ratio W/H compared against per-pin baseline
    │  "Fallen" confirmed after FALL_CONFIRM consecutive flagged frames
    │  Fallen pins anchored spatially to prevent ID drift
    ▼
Frame annotation
    │  Green box  →  fallen pin (numbered in fall order)
    │  Red box    →  standing pin
    ▼
Output video  +  2-second summary screen
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 detection & BoTSort tracking |
| `opencv-python` | Video I/O and frame annotation |
| `scikit-learn` | Train/val/test split generation |
| `numpy` | Numerical operations |

---

## 🎓 Model Training

To retrain on new data:

1. Annotate images using [Roboflow](https://roboflow.com) and export in YOLOv8 format
2. Place the export under `dataset/`
3. Open `notebook/bowling_detection.ipynb`
4. Update the `source_dataset` path in Cell 3
5. Run Cells 1–9 to split the data and train
6. Best weights will be saved to `runs/detect/bowling_pin_only/weights/best.pt`

---

## 🐛 Troubleshooting

**Video not opening**
```
ValueError: Could not open video
```
Check that the file path is correct and the format is supported (`.mp4`, `.MOV`, `.avi`).

**GPU out of memory**  
Reduce batch size in the training cell:
```python
model.train(data=yaml_path, epochs=50, batch=2, ...)
```

**Output video has no audio**  
OpenCV's `VideoWriter` does not copy audio. Use `ffmpeg` to mux audio back if needed:
```bash
ffmpeg -i output_pins_only.mp4 -i input.MOV -c copy -map 0:v -map 1:a final.mp4
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📝 License

Dataset licensed privately via Roboflow (workspace: `youssif-workspace-eywqn`). Code is available for educational and research purposes.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 framework  
- [OpenCV](https://opencv.org/) — Computer vision library  
- [Roboflow](https://roboflow.com/) — Dataset annotation and management  
- [BoTSort](https://github.com/NirAharon/BoT-SORT) — Multi-object tracking algorithm  

---