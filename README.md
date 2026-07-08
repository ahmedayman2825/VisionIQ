<div align="center">

# 👁️ VisionIQ

### Real-time Face Age & Gender Estimation

Detects faces in images or video and predicts **age bracket** and **gender** using a lightweight CNN built on MobileNetV3-Small, paired with a YOLOv8x face detector.

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

VisionIQ is a two-stage computer vision pipeline for real-time face analysis:

1. A **YOLOv8x** face detector (`yolov8x_person_face.pt`) locates faces in a frame.
2. Each detected face is cropped and passed to a custom **dual-head classifier** (MobileNetV3-Small backbone) that predicts:
   - **Age bracket** — 9 classes
   - **Gender** — Male / Female

The classifier is trained on the **[FairFace dataset](https://www.kaggle.com/datasets/aibloy/fairface)**, a face dataset balanced across race, gender, and age for fairer predictions.

```
Frame → YOLOv8x (face detection) → Crop face → Grayscale + Resize (224×224)
      → VisionIQ model → { age bracket, gender } → Draw label on frame
```

---

## ✨ Features

- 🎯 Real-time face detection using YOLOv8x
- 🧑‍🦱 Joint age-bracket + gender prediction from a single lightweight model
- 🎥 Live webcam demo with on-screen bounding boxes and labels
- 🪶 Lightweight backbone (MobileNetV3-Small) — runs on CPU or GPU
- 🔬 Trained on FairFace for more balanced predictions across demographics

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, torchvision |
| Face Detection | Ultralytics YOLOv8x |
| Image/Video I/O | OpenCV, Pillow |
| Data Handling | pandas |
| Training Utilities | tqdm |

---

## 🏗️ Model Architecture

| Component | Detail |
|---|---|
| Backbone | MobileNetV3-Small (ImageNet pretrained, first conv modified to accept 1-channel grayscale input) |
| Input | 224 × 224, grayscale, normalized (mean=0.5, std=0.5) |
| Age head | Linear(576→256) → ReLU → Dropout(0.3) → Linear(256→9) |
| Gender head | Linear(576→128) → ReLU → Dropout(0.3) → Linear(128→2) |
| Loss | CrossEntropyLoss (age) + CrossEntropyLoss (gender), summed |
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4, ReduceLROnPlateau scheduler |

**Age brackets:** `0-2` · `3-9` · `10-19` · `20-29` · `30-39` · `40-49` · `50-59` · `60-69` · `70+`

### Validation Accuracy

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

Download `best_model.pth` from the **[Releases page](https://github.com/ahmedayman2825/VisionIQ/releases)** and place it at:

```
models/best_model.pth
```

You'll also need the YOLOv8x face detector weights at:

```
models/yolov8x_person_face.pt
```

### Get the dataset (for training only)

If you want to retrain the model, download **[FairFace from Kaggle](https://www.kaggle.com/datasets/aibloy/fairface)** and place it at:

```
dataset/FairFace/
```

---

## 🚀 Usage

### Live webcam demo

```bash
cd src
python test_cam.py
```

Press **ESC** to quit. Each detected face is boxed with a `Gender | Age` label.

### Training from scratch

```bash
cd src
python train.py
```

### Using the model in your own code

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import VisionIQ

AGE_CLASSES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER_CLASSES = ["Male", "Female"]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionIQ().to(device)
checkpoint = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

face = Image.open("face_crop.jpg")  # already-cropped face image
tensor = transform(face).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(tensor)
    age = AGE_CLASSES[outputs["age"].argmax(1).item()]
    gender = GENDER_CLASSES[outputs["gender"].argmax(1).item()]

print(f"{gender}, {age}")
```

---

## 📁 Folder Structure

```
VisionIQ/
├── src/
│   ├── model.py               # VisionIQ architecture (backbone + age/gender heads)
│   ├── dataset.py             # FairFace Dataset class + transforms
│   ├── fix_fairface_labels.py # Label preprocessing helper
│   ├── train.py               # Training loop
│   ├── test_cam.py            # Live webcam inference demo
│   └── webcam.py
├── models/                    # best_model.pth + yolov8x_person_face.pt (not tracked)
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
- [ ] Export to ONNX for faster CPU inference
- [ ] Improve age accuracy with a stronger backbone or ordinal regression loss
- [ ] Add a batch image/video-file inference script (currently webcam-only)
- [ ] Add a proper evaluation script with confusion matrices per class
