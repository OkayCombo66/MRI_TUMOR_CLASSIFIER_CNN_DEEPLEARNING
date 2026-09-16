"""Model definitions: a small custom CNN, and a ResNet-18 baseline."""

import torch.nn as nn
import torchvision.models as tv


class CNN(nn.Module):
    """A small 4-block CNN for 128x128 input images."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),  # single logit (binary classification)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def resnet18_binary(in_channels=3, pretrained=True):
    """ResNet-18 adapted for binary classification.

    Set in_channels=1 to swap the first conv layer for single-channel
    (grayscale) input; ImageNet pretrained weights are only meaningful
    for the default in_channels=3 case.
    """
    weights = tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = tv.resnet18(weights=weights)
    if in_channels == 1:
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, 1)  # logits
    return m


def build_model(name="cnn", in_channels=3, pretrained=True):
    """Small factory so scripts can pick a model by name from config."""
    name = name.lower()
    if name == "cnn":
        return CNN(in_channels=in_channels)
    if name in {"resnet18", "resnet"}:
        return resnet18_binary(in_channels=in_channels, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {name!r} (expected 'cnn' or 'resnet18')")
