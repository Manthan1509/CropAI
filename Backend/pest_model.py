import torch
import json
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import io

# Load Class Indices
with open("Backend/pest_model/pest_classes.json", "r") as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(pretrained=False)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_names)),
    nn.Softmax(dim=1)
)

model.load_state_dict(torch.load("Backend/pest_model/pest_model.pth", map_location=device))
model.to(device)
model.eval()

# Function to Predict Image from Bytes
def predict_pest(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return {"pest": class_names[predicted.item()]}
