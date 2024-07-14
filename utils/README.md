### 1.extract.py에 필요한 모듈 설치
아나콘다로 가상환경에서 설치 추천드립니다
```python
pip install opencv-python numpy python-pptx python-docx PyMuPDF pillow pdf2docx paddlepaddle paddleocr
```
-----------------------------------------------
### 2.extract.py 
extract_images_from_pptx(pptx_path, output_folder):

extract_images_from_docx(docx_path, output_folder)

extract_images_from_pdf(pdf_path, output_folder)

입력: 파일주소, 생성될 이미지 폴더


-> 파일로부터 이미지를 추출하여 폴더가 생성됨

-> 폴더 생성안하고 os모듈을 이용하여 코드상에서 받아서 처리해도됨 (현재는 디버깅 떄문)

-----------------------------------------------

extract_image_from_text(figure_path)

입력: 모델로부터 생성된 figure 폴더

출력: figure안에 있는 텍스트

-----------------------------------------------
### 3.classify.py
classify_chart_images(model, image_folder, figure_folder, device)

입력: 모델, 추출된 이미지 폴더, 분류될 figure 폴더, device(cpu인지 gpu인지)

-----------------------------------------------
### 4.사용 순서도 
```python
1.모듈 임포트 및 모델 로드
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 3 
model = EfficientNet.from_pretrained('efficientnet-b4')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 저장된 모델 가중치를 CPU로 로드
model_weights_path = "/Users/hwany/AI-corning/classification/efficientnet_finetuned.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))  # CPU로 가중치 매핑
model.eval()
```
모델 다운로드 링크: 
https://huggingface.co/hwan99/corning-figure-classifier/tree/main


```python
2.코드 조각
from utils.classify import classify_chart_images
from utils.extract import *

#1. pdf => image 추출 
pdf_path = ""
image_folder = ""
extract_images_from_pdf(pdf_path, image_folder)

#2. image => figure 분류 
figure_folder = ""
classify_chart_images(model, image_folder, figure_folder, device)

#3. figure => text 추출
figure_path = ""
figure_text = extract_images_from_text(figure_path)
```







