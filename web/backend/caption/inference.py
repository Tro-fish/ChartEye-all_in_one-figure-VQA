from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import re
import io
import torch
import base64
from django.conf import settings

# device
device = settings.DEVICE

# model
dtype = torch.bfloat16
model_id = "google/paligemma-3b-ft-scicap-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    revision="bfloat16",
).eval()

processor = AutoProcessor.from_pretrained(model_id)

def post_process(text):
    # Fixing spaces before punctuations
    text = re.sub(r'\s+([.,!?()])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    
    # Capitalizing the first letter of the sentence and 'I' pronouns
    sentences = re.split('(?<=[.!?]) +', text)
    sentences = [s.capitalize() for s in sentences]
    processed_text = ' '.join(sentences)
    
    return processed_text

# Instruct the model to create a caption in Spanish
def captioning(img_str):
    global device, model, processor

    prompt = ""
    image = base64.b64decode(img_str)
    image = Image.open(io.BytesIO(image)).convert("RGB")

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    print('Start Captioning...')
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    print('End Captioning...')
    
    return post_process(decoded)

