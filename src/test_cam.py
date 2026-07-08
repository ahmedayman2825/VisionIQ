import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms

from model import VisionIQ

# =====================================================
# Settings
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIDENCE = 0.50

AGE_CLASSES = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]

GENDER_CLASSES = [
    "Male",
    "Female"
]

# =====================================================
# Load VisionIQ
# =====================================================

visioniq = VisionIQ().to(DEVICE)

checkpoint = torch.load(
    "models/best_model.pth",
    map_location=DEVICE
)

visioniq.load_state_dict(checkpoint["model"])
visioniq.eval()

# =====================================================
# Load YOLO Face Detector
# =====================================================

detector = YOLO("models/yolov8x_person_face.pt")

# =====================================================
# Image Transform
# =====================================================

transform = transforms.Compose([

    transforms.Grayscale(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.5], [0.5])

])

# =====================================================
# Webcam
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Couldn't open webcam.")
    exit()

print("Press ESC to quit.")

# =====================================================
# Loop
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = detector.predict(
        frame,
        imgsz=1280,
        conf=0.25,
        verbose=False
    )

    boxes = results[0].boxes

    face_found = False

    for box in boxes:

        cls = int(box.cls.item())

        # Ignore persons
        if cls != 1:
            continue

        conf = float(box.conf.item())

        if conf < CONFIDENCE:
            continue

        face_found = True

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            outputs = visioniq(tensor)

            age = AGE_CLASSES[
                outputs["age"].argmax(1).item()
            ]

            gender = GENDER_CLASSES[
                outputs["gender"].argmax(1).item()
            ]

        label = f"{gender} | {age}"

        # ----------------------------
        # Draw Box
        # ----------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - th - 10),
            (x1 + tw + 10, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    if not face_found:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("VisionIQ", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()