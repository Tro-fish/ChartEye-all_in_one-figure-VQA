import os
import shutil
from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  
    image = transform(image).unsqueeze(0)  
    return image

# 단일 이미지 예측 함수
def classify_image(model, image_path, device):
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        idx_to_class = {0: 'figure', 1: 'table', 2: 'trash'}  
        predicted_label = idx_to_class[predicted.item()]
    
    return predicted_label

# 추출된 이미지 전체 분류 함수 
def classify_chart_images(model, image_folder, figure_folder, device):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    idx_to_class = {0: 'figure', 1: 'table', 2: 'trash'}
    
    for image_name in os.listdir(image_folder):
        print(f'Classifying {image_name}...')
        image_path = os.path.join(image_folder, image_name)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        image_tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = idx_to_class[predicted.item()]
            print(predicted_label)
            if predicted_label == 'figure':
                shutil.copy(image_path, os.path.join(figure_folder, image_name))
