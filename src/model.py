import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class VisionIQ(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        # -------------------------------
        # Change first layer to 1 channel
        # -------------------------------

        old_conv = self.backbone.features[0][0]

        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize the new conv weights
        with torch.no_grad():
            self.backbone.features[0][0].weight[:] = (
                old_conv.weight.mean(dim=1, keepdim=True)
            )

        # Remove original classifier
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        # -------------------------------
        # Heads
        # -------------------------------

        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 9)      # 9 FairFace age classes
        )

        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        features = self.backbone(x)

        age = self.age_head(features)
        gender = self.gender_head(features)

        return {
            "age": age,
            "gender": gender
        }