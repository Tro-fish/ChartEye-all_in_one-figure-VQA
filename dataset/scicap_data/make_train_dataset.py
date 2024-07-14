import json
from tqdm import tqdm

with open('/home/wani/Desktop/ChartQA/dataset/datasets--CrowdAILab--scicap/snapshots/60e504baa94423f63cda87d5442e73a696b953d3/train-acl.json', 'r') as f:
    train_data = json.load(f)

train_data = train_data['annotations']
final_data = []

for data in tqdm(train_data, total=len(train_data)):
    final_data.append({"image_id":data['image_id'],'caption_id':data['id'],'caption': data['caption_no_index']})

with open('/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/train/final_train.json', 'w') as f:
    json.dump(final_data, f, indent=4)