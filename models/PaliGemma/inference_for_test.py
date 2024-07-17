
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import requests
import torch
import json

model_id = "google/paligemma-3b-ft-scicap-448"
device = "cuda:1"
prompt = "caption en\n"
dtype = torch.bfloat16
image_path_prefix = "/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/images/"

with open ("/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/final_validation_v2.json", 'r') as f:
    validation_data = json.load(f)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()

processor = AutoProcessor.from_pretrained(model_id)
result = []
for data in tqdm(validation_data, total=len(validation_data)):
    image_path = image_path_prefix + ("00000" + str(data['image_id']) + '.png')
    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
        result.append({"predition":decoded, "gold":data['caption']})
        with open('/home/wani/Desktop/Corning_team3/models/PaliGemma/paligemma-3b-ft-scicap-448_validation.json', 'w') as f:
            json.dump(result, f, indent=4)