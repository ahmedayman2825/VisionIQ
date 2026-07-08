<div align="center">

# 👁️ VisionIQ

### Real-time Face Age, Gender & Emotion Estimation

Detects faces in images or video and predicts **age bracket**, **gender**, and **emotion** using a YOLOv8x face detector paired with a custom MobileNetV3-Small age/gender classifier and a pretrained emotion model.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[![Download Model](https://img.shields.io/badge/Download-Model%20Weights-success?style=for-the-badge&logo=github)](https://github.com/ahmedayman2825/VisionIQ/releases)

[Overview](#-overview) • [Features](#-features) • [Tech Stack](#-tech-stack) • [Installation](#-installation) • [Usage](#-usage) • [Structure](#-folder-structure) • [Contributors](#-contributors) • [Future Improvements](#-future-improvements)

</div>

---

## 📌 Overview

VisionIQ is a real-time face analysis pipeline with three stages:

1. **`FaceDetector`** (YOLOv8x, `yolov8x_person_face.pt`) locates faces in a frame.
2. **`AgeGenderPredictor`** — a custom dual-head classifier (MobileNetV3-Small backbone) trained on **[FairFace](https://www.kaggle.com/datasets/aibloy/fairface)** — predicts age bracket and gender from each face crop.
3. **`EmotionPredictor`** — a pretrained EfficientNet-B0 model (8-class, trained on AFEW) — predicts the emotion expressed on each face.

```
Frame → YOLOv8x (face detection) → Crop face
      → AgeGenderPredictor → { age bracket, gender }
      → EmotionPredictor   → { emotion, confidence }
      → Draw all labels on frame
```

`main.py` wires all three together into a live webcam demo.

---

## ✨ Features

- 🎯 Real-time face detection using YOLOv8x
- 🧑‍🦱 Age-bracket + gender prediction from a custom lightweight model
- 😀 Emotion recognition (8 classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise)
- 🎥 Live webcam demo with on-screen bounding boxes and stacked labels
- 🧩 Modular design — detector, age/gender, and emotion are separate, swappable components
- 🪶 Lightweight age/gender backbone (MobileNetV3-Small) — runs on CPU or GPU

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, torchvision |
| Face Detection | Ultralytics YOLOv8x |
| Emotion Recognition | EfficientNet-B0 (pretrained, [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)) |
| Image/Video I/O | OpenCV, Pillow |
| Data Handling | pandas |
| Training Utilities | tqdm |

---

## 🏗️ Age/Gender Model Architecture

| Component | Detail |
|---|---|
| Backbone | MobileNetV3-Small (ImageNet pretrained, first conv modified to accept 1-channel grayscale input) |
| Input | 224 × 224, grayscale, normalized (mean=0.5, std=0.5) |
| Age head | Linear(576→256) → ReLU → Dropout(0.3) → Linear(256→9) |
| Gender head | Linear(576→128) → ReLU → Dropout(0.3) → Linear(128→2) |
| Loss | CrossEntropyLoss (age) + CrossEntropyLoss (gender), summed |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4, ReduceLROnPlateau scheduler |

**Age brackets:** `0-2` · `3-9` · `10-19` · `20-29` · `30-39` · `40-49` · `50-59` · `60-69` · `70+`

Emotion recognition uses a separate, pretrained EfficientNet-B0 model (not trained in this repo) that classifies 8 emotions directly from an RGB face crop resized to 224×224.

### Validation Accuracy (Age/Gender model)

| Task | Accuracy |
|---|---|
| Age (9-class) | **57.14%** |
| Gender (binary) | **91.82%** |

> Age is a hard 9-way problem — adjacent brackets (e.g. `20-29` vs `30-39`) are easy to confuse — so 57% is a reasonable result for a lightweight backbone. Gender accuracy is strong at 91.82%.

---

## 📦 Installation

```bash
git clone https://github.com/ahmedayman2825/VisionIQ.git
cd VisionIQ
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch, torchvision, ultralytics, OpenCV, Pillow, pandas.

### Get the model weights

Download these from the **[Releases page](https://github.com/ahmedayman2825/VisionIQ/releases)** and place them in `models/` — see [`models/README.md`](models/README.md) for details:

```
models/
├── best_model.pth
├── yolov8x_person_face.pt
└── enet_b0_8_best_afew.pt
```

### Get the dataset (for training only)

If you want to retrain the age/gender model, download **[FairFace from Kaggle](https://www.kaggle.com/datasets/aibloy/fairface)** — see [`dataset/README.md`](dataset/README.md) for details.

---

## 🚀 Usage

### Live webcam demo (age, gender, and emotion)

```bash
cd src
python main.py
```

Press **ESC** to quit. Each detected face is boxed, showing gender, age bracket, and predicted emotion with confidence.

### Training the age/gender model from scratch

```bash
cd src
python train.py
```

### Using the components in your own code

```python
import cv2
from detector import FaceDetector
from age_gender import AgeGenderPredictor
from emotion import EmotionPredictor

detector = FaceDetector()
age_gender = AgeGenderPredictor()
emotion = EmotionPredictor()

frame = cv2.imread("photo.jpg")
faces = detector.detect(frame)

for detected in faces:
    face = detected["face"]
    ag = age_gender.predict(face)
    em = emotion.predict(face)
    print(ag["gender"], ag["age"], em["emotion"], em["confidence"])
```

---

## 📁 Folder Structure

```
VisionIQ/
├── src/
│   ├── model.py               # VisionIQ age/gender architecture (backbone + heads)
│   ├── dataset.py             # FairFace Dataset class + transforms
│   ├── fix_fairface_labels.py # Label preprocessing helper
│   ├── train.py               # Training loop (age/gender model)
│   ├── detector.py            # FaceDetector — YOLOv8x face detection
│   ├── age_gender.py          # AgeGenderPredictor — inference wrapper
│   ├── emotion.py             # EmotionPredictor — emotion inference wrapper
│   └── main.py                # Live webcam demo combining all three
├── models/                    # best_model.pth, yolov8x_person_face.pt, enet_b0_8_best_afew.pt (not tracked)
├── dataset/                   # FairFace dataset (not tracked)
├── requirements.txt
└── README.md
```

---

## 👤 Contributors

 **Ahmed Ayman**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eng-ahmed-ayman/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ahmedayman2825)

---

## 🗺️ Future Improvements

- [ ] Log training metrics to a file/CSV instead of console only
- [ ] Export models to ONNX for faster CPU inference
- [ ] Improve age accuracy with a stronger backbone or ordinal regression loss
- [ ] Fine-tune the emotion model on more diverse, in-the-wild data
- [ ] Add a batch image/video-file inference script (currently webcam-only)
- [ ] Add a proper evaluation script with confusion matrices per class
