import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms

from model import VisionIQ


class AgeGenderPredictor:

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

    def __init__(
        self,
        model_path="models/best_model.pth",
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = VisionIQ().to(self.device)

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint["model"])

        self.model.eval()

        self.transform = transforms.Compose([

            transforms.Grayscale(),

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize([0.5], [0.5])

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

            outputs = self.model(tensor)

            age = self.AGE_CLASSES[
                outputs["age"].argmax(1).item()
            ]

            gender = self.GENDER_CLASSES[
                outputs["gender"].argmax(1).item()
            ]

        return {
            "age": age,
            "gender": gender
        }