import torch
import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=4, pretrained=True):
    """
    Returns a ResNet18 model pre-trained on ImageNet,
    with the final fully-connected layer replaced for `num_classes`.
    """
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_vgg16(num_classes=4, pretrained=True):
    """
    Returns a VGG16 model pre-trained on ImageNet,
    with the classifier adjusted for `num_classes`.
    """
    model = models.vgg16(pretrained=pretrained)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x