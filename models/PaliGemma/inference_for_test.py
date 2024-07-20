
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import torch
import json
import pandas as pd

# model
model_id = "google/paligemma-3b-ft-scicap-448"
device = torch.device("cuda")
prompt = "caption en\n"
dtype = torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()

processor = AutoProcessor.from_pretrained(model_id)

# image path prefix
image_path_prefix1 = "../../dataset/eval/images/"
image_path_prefix2 = "../../dataset/eval/cm_images/"

# data
with open ("../../dataset/eval/final_validation_v2.json", 'r') as f:
    validation_data1 = json.load(f)
    validation_data1 = pd.DataFrame(validation_data1)
validation_data2 = pd.read_csv("../../dataset/eval/chem-mate-png.csv")
validation_data = pd.concat([validation_data1, validation_data2], ignore_index=True)

# inference
result = []
except_list = []
for i, data in tqdm(validation_data.iterrows(), total=len(validation_data)):
    image_id = str(data['image_id'])
    if '.png' in image_id:
        image_path = image_path_prefix2 + image_id
    else:
        image_path = image_path_prefix1 + ("00000" + image_id + '.png')

    try:
        image = Image.open(image_path).convert("RGB")
        model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
            generation = generation[0][input_len:]
            decoded = processor.decode(generation, skip_special_tokens=True)
            gold = ' '.join(str(data['caption']).split())
            result.append({"predition":decoded, "gold":gold})
            with open('../../dataset/eval/validation.json', 'w') as f:
                json.dump(result, f, indent=4)
    except:
        except_list.append(image_id)
print(except_list)