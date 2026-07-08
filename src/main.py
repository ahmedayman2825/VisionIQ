import cv2

from detector import FaceDetector
from age_gender import AgeGenderPredictor
from emotion import EmotionPredictor


# =====================================================
# Load Models
# =====================================================

detector = FaceDetector()

age_gender = AgeGenderPredictor()

emotion = EmotionPredictor()

# =====================================================
# Webcam
# =====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Couldn't open webcam.")
    exit()

print("Press ESC to quit.")

# =====================================================
# Main Loop
# =====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = detector.detect(frame)

    if len(faces) == 0:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    for detected in faces:

        face = detected["face"]

        bbox = detected["bbox"]

        x1, y1, x2, y2 = bbox

        # -------------------------
        # Age + Gender
        # -------------------------

        ag = age_gender.predict(face)

        # -------------------------
        # Emotion
        # -------------------------

        em = emotion.predict(face)

        detector.draw_box(frame, bbox)

        # -------------------------
        # Labels
        # -------------------------

        line1 = ag["gender"]

        line2 = ag["age"]

        line3 = f'{em["emotion"]} ({em["confidence"]*100:.1f}%)'

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.7

        thickness = 2

        (w1, h1), _ = cv2.getTextSize(
            line1,
            font,
            scale,
            thickness,
        )

        (w2, h2), _ = cv2.getTextSize(
            line2,
            font,
            scale,
            thickness,
        )

        (w3, h3), _ = cv2.getTextSize(
            line3,
            font,
            scale,
            thickness,
        )

        width = max(w1, w2, w3) + 20

        height = h1 + h2 + h3 + 30

        top = max(0, y1 - height)

        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + width, y1),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame,
            line1,
            (x1 + 8, top + h1 + 3),
            font,
            scale,
            (0, 0, 0),
            thickness,
        )

        cv2.putText(
            frame,
            line2,
            (x1 + 8, top + h1 + h2 + 12),
            font,
            scale,
            (0, 0, 0),
            thickness,
        )

        cv2.putText(
            frame,
            line3,
            (x1 + 8, top + h1 + h2 + h3 + 20),
            font,
            scale,
            (0, 0, 0),
            thickness,
        )

    cv2.imshow("VisionIQ", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()