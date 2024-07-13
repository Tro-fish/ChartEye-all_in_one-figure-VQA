import json
import os
import shutil
from tqdm import tqdm

# JSON 파일 경로와 이미지 폴더 경로
json_file_path = '/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/final_validation_v2.json'
images_folder_path = '/home/wani/Desktop/ChartQA/dataset/datasets--CrowdAILab--scicap/snapshots/60e504baa94423f63cda87d5442e73a696b953d3/share-task-img-mask/arxiv/val'
destination_folder_path = '/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/images'

# JSON 데이터 로드
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# 필요한 이미지들을 destination 폴더로 복사
for data in tqdm(json_data, total = len(json_data)):
    image_id = data['image_id']
    image_filename = f"{image_id:012d}.png"
    source_path = os.path.join(images_folder_path, image_filename)
    destination_path = os.path.join(destination_folder_path, image_filename)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied {image_filename} to {destination_folder_path}")
    else:
        print(f"Image {image_filename} not found in {images_folder_path}")

print(len(json_data))
print("Image extraction completed.")