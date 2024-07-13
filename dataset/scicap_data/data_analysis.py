import json
from tqdm import tqdm

with open('/home/wani/Desktop/ChartQA/dataset/scicap_data/val.json','r') as f:
    json_data = json.load(f)

figure_type = set()

print(json_data.keys()) # images, annotations

for image in tqdm(json_data['images'], total = len(json_data)):
    figure_type.add(image['figure_type'])

print(figure_type)