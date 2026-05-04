import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, VGG16_Weights


def get_resnet18(num_classes: int = 4, pretrained: bool = True) -> nn.Module:

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_vgg16(num_classes: int = 4, pretrained: bool = True) -> nn.Module:

    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)

    in_features = model.classifier[6].in_features
    model.classifier[5] = nn.Dropout(p=0.5)
    model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


def get_model(architecture: str, num_classes: int = 4, pretrained: bool = True) -> nn.Module:

    architecture = architecture.lower()
    if architecture == 'resnet18':
        return get_resnet18(num_classes=num_classes, pretrained=pretrained)
    elif architecture == 'vgg16':
        return get_vgg16(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Choose from: 'resnet18', 'vgg16'."
        )
