import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')

# model
num_classes = 3 
model = EfficientNet.from_pretrained('efficientnet-b4')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

model_weights_path = "../model/efficientnet_finetuned.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  

    return image

def classify_chart_images(images):
    global model, device
    idx_to_class = {0: 'figure', 1: 'table', 2: 'trash'}
    
    figures = []
    for image in tqdm(images, desc='Classifying...'):
        image_tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = idx_to_class[predicted.item()]
            if predicted_label == 'figure':
                figures.append(image)
    
    return figures
