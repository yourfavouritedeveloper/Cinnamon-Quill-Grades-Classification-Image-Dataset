

import argparse
import torch
from PIL import Image
from torchvision import transforms

from models.cinnamon_model import get_model

CLASSES     = ["Alba", "C4", "C5", "C5 Special"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(arch: str, checkpoint_path: str) -> torch.nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("state_dict", checkpoint)

    model = get_model(arch, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def predict(image_path: str, model: torch.nn.Module, device: torch.device) -> str:

    img = Image.open(image_path).convert("RGB")
    tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        _, predicted = output.max(1)

    return CLASSES[predicted.item()]


def main():
    parser = argparse.ArgumentParser(description="Cinnamon quill grade predictor")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--model",      required=True, choices=["resnet18", "vgg16"],
                        help="Model architecture (must match checkpoint)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file")
    args = parser.parse_args()

    model, device = load_model(args.model, args.checkpoint)
    grade = predict(args.image, model, device)

    print("\n===================================")
    print(f"  Image:        {args.image}")
    print(f"  Architecture: {args.model}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Predicted Grade: {grade}")
    print("===================================\n")


if __name__ == "__main__":
    main()
