import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms


class EmotionPredictor:

    EMOTIONS = [
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise"
    ]

    def __init__(
        self,
        model_path="models/enet_b0_8_best_afew.pt",
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

        ])

    def predict(self, face):

        rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():

            output = self.model(tensor)

            probs = torch.softmax(
                output,
                dim=1
            )[0]

        emotion_idx = torch.argmax(probs).item()

        emotion = self.EMOTIONS[
            emotion_idx
        ]

        confidence = probs[
            emotion_idx
        ].item()

        return {
            "emotion": emotion,
            "confidence": confidence
        }

    def predict_top3(self, face):

        rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():

            output = self.model(tensor)

            probs = torch.softmax(
                output,
                dim=1
            )[0]

        values, indices = torch.topk(
            probs,
            3
        )

        results = []

        for value, index in zip(values, indices):

            results.append({

                "emotion": self.EMOTIONS[
                    index.item()
                ],

                "confidence": value.item()

            })

        return results