import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 8):
    model = models.resnet50(weights="DEFAULT")
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model
