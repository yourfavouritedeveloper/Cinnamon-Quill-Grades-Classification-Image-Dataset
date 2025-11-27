import torch
from PIL import Image
from torchvision import transforms, models

num_classes = 4
classes = ['Alba', 'C4', 'C5', 'C5 Special']   

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("checkpoints/saved_model_pretrained.pth", map_location="cpu") #change checkpoint
model.load_state_dict(checkpoint['state_dict'])
model.eval()

image_path = r"datasets\cinnamon\C5 Special\C5 Special 08.JPG"

img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    _, predicted = output.max(1)

print("\n===================================")
print(" Image:", image_path)
print(" Predicted Grade:", classes[predicted.item()])
print("===================================\n")
