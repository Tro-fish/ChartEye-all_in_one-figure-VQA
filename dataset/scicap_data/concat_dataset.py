import json
from tqdm import tqdm
with open('/home/wani/Desktop/ChartQA/dataset/datasets--CrowdAILab--scicap/snapshots/60e504baa94423f63cda87d5442e73a696b953d3/val-metadata.json', 'r') as f:
    val_meta_data = json.load(f)
with open('/home/wani/Desktop/ChartQA/dataset/datasets--CrowdAILab--scicap/snapshots/60e504baa94423f63cda87d5442e73a696b953d3/val.json', 'r') as f:
    val_data = json.load(f)

with open('/home/wani/Desktop/ChartQA/dataset/datasets--CrowdAILab--scicap/snapshots/60e504baa94423f63cda87d5442e73a696b953d3/test-metadata.json', 'r') as f:
    test_meta_data = json.load(f)
with open('/home/wani/Desktop/ChartQA/dataset/datasets--CrowdAILab--scicap/snapshots/60e504baa94423f63cda87d5442e73a696b953d3/public-test.json', 'r') as f:
    test_data = json.load(f)

final_data = []
val_data = val_data['annotations']
for data in tqdm(val_meta_data,total=len(val_meta_data)):
    image_id = data['image_id']
    for val in val_data:
        if val['image_id'] == image_id:
            final_data.append({"categories":data['categories'],"image_id":val['image_id'],'caption_id':val['id'],'caption': val['caption_no_index'], 'arXiv_id':data['arXiv_id']})

"""
test_data = test_data['annotations']
for data in tqdm(test_meta_data,total=len(test_meta_data)):
    image_id = data['image_id']
    for test in test_data:
        if test['image_id'] == image_id:
            final_data.append({"categories":data['categories'],"image_id":test['image_id'],'caption_id':test['id'],'caption': test['caption_no_index']})
"""

with open('/home/wani/Desktop/ChartQA/dataset/scicap_data/concated_val.json', 'w') as f:
    json.dump(final_data, f, indent=4)