from pathlib import Path

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


AGE_TO_CLASS = {
    "0-2": 0,
    "3-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
    "70+": 8,
}


GENDER_TO_CLASS = {
    "Male": 0,
    "Female": 1,
    0: 0,
    1: 1,
}


class FairFaceDataset(Dataset):

    def __init__(self, csv_file, root_dir, train=True):

        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)

        if train:

            self.transform = transforms.Compose([

                transforms.Grayscale(),

                transforms.Resize((224, 224)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomRotation(10),

                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )

            ])

        else:

            self.transform = transforms.Compose([

                transforms.Grayscale(),

                transforms.Resize((224, 224)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )

            ])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.root_dir / row["file"]

        image = Image.open(image_path)

        image = self.transform(image)

        age = AGE_TO_CLASS[row["age"]]

        gender = GENDER_TO_CLASS[row["gender"]]

        return image, age, gender