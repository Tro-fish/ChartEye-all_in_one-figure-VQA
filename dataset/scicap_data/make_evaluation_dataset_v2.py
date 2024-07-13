import json

# 상위 카테고리에 대한 접두어 정의
category_mapping = {
    "Computer Science": ["cs"],
    "Electrical Engineering and Systems Science": ["eess"],
    "Mathematics": ["math"],
    "Physics": ["astro-ph", "cond-mat", "physics", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th", "nucl-ex", "nucl-th", "quant-ph"],
    "Statistics": ["stat"]
}

def classify_category(category):
    for main_category, prefixes in category_mapping.items():
        if len(CS) >= 1000 and main_category=="Computer Science":
            continue
        elif len(EES) >= 1000 and main_category=="Electrical Engineering and Systems Science":
            continue
        elif len(Math) >= 1000 and main_category=="Mathematics":
            continue
        elif len(Statistics) >= 1000 and main_category=="Statistics":
            continue
        for prefix in prefixes:
            if prefix in category:
                return main_category
    return "Unknown"

with open('/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/concated_validation.json', 'r') as f:
    json_data = json.load(f)

CS = []
EES = []
Math = []
Physics = []
Statistics = []
final = []

# 카테고리별로 데이터를 분류하여 저장
for data in json_data:
    category = data['categories']
   
    main_category = classify_category(category)
    if main_category == "Computer Science" and len(CS) < 1000:
        CS.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption']})
        final.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption'], 'category': main_category, "arXiv_id": data['arXiv_id'] })

    elif main_category == "Electrical Engineering and Systems Science" and len(EES) < 1000:
        EES.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption']})
        final.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption'], 'category': main_category, "arXiv_id": data['arXiv_id'] })

    elif main_category == "Mathematics" and len(Math) < 1000:
        Math.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption']})
        final.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption'], 'category': main_category, "arXiv_id": data['arXiv_id'] })

    elif main_category == "Physics" and len(Physics) < 1000:
        Physics.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption']})
        final.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption'], 'category': main_category, "arXiv_id": data['arXiv_id'] })

    elif main_category == "Statistics" and len(Statistics) < 1000:
        Statistics.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption']})
        final.append({'image_id': data['image_id'], 'caption_id': data['caption_id'], 'caption': data['caption'], 'category': main_category, "arXiv_id": data['arXiv_id'] })

    


# 각 카테고리 배열의 크기를 출력
print(f"Computer Science: {len(CS)}")
print(f"Electrical Engineering and Systems Science: {len(EES)}")
print(f"Mathematics: {len(Math)}")
print(f"Physics: {len(Physics)}")
print(f"Statistics: {len(Statistics)}")


# 최종 결과를 JSON 형식으로 저장
result = {
    "Computer Science": CS,
    "Electrical Engineering and Systems Science": EES,
    "Mathematics": Math,
    "Physics": Physics,
    "Statistics": Statistics
}

with open('/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/categorized_validation_v2.json', 'w') as f:
    json.dump(result, f, indent=4)

with open('/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/final_validation_v2.json', 'w') as f:
    json.dump(final, f, indent=4)