import cv2
from ultralytics import YOLO


class FaceDetector:
    def __init__(
        self,
        model_path="models/yolov8x_person_face.pt",
        conf=0.50,
        margin=30,
        imgsz=1280,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.margin = margin
        self.imgsz = imgsz

    def detect(self, frame):
        """
        Returns:
        [
            {
                "bbox": (x1, y1, x2, y2),
                "face": face_crop,
                "confidence": confidence
            },
            ...
        ]
        """

        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=0.25,
            verbose=False,
        )

        detections = []

        h, w = frame.shape[:2]

        if len(results) == 0:
            return detections

        boxes = results[0].boxes

        for box in boxes:

            cls = int(box.cls.item())

            # class 1 = face
            if cls != 1:
                continue

            confidence = float(box.conf.item())

            if confidence < self.conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Add margin
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(w, x2 + self.margin)
            y2 = min(h, y2 + self.margin)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "face": face,
                    "confidence": confidence,
                }
            )

        return detections

    @staticmethod
    def draw_box(frame, bbox, color=(0, 255, 0), thickness=2):

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )