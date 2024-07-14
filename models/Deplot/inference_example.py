from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image
import torch

# 모델과 프로세서를 로드하고 GPU로 이동
processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to('cuda')

# 이미지를 가져와서 로드
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/5090.png"
url2 = "/home/wani/Desktop/Corning_team3/test_images/image3.png"
image = Image.open(requests.get(url2, stream=True).raw)

# 입력을 처리하고 GPU로 이동
inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
inputs = {k: v.to('cuda') for k, v in inputs.items()}

# 예측을 생성
predictions = model.generate(**inputs, max_new_tokens=512)

# 결과를 디코딩하여 출력
print(processor.decode(predictions[0], skip_special_tokens=True))


"""
Entity | Individuals responsibility | Government's responsibility <0x0A> MEDIAN | 39.0 | 55.0 <0x0A> Germany | nan | 35.0 <0x0A> UK | 45.0 | 49.0 <0x0A> Sweden | 37.0 | 53.0 <0x0A> Denmark | 42.0 | 54.0 <0x0A> France | 38.0 | 55.0 <0x0A> Netherla nns | 40.0 | 58.0 <0x0A> Spain | 29.0 | 63.0 <0x0A> Italy | 22.0 | 74.0
"""