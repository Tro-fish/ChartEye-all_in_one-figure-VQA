from django.http import JsonResponse
import json
import openai
import torch
import base64
import io
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from dotenv import dotenv_values
from django.conf import settings

config = dotenv_values("../.env")
client = openai.OpenAI(api_key=config.get("API_KEY"))
device = settings.DEVICE

model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image = data.get('image')
        text = data.get('text')
        caption = data.get('caption')

        answer = get_answer_llava(image, text, caption)
        return JsonResponse({'status': 'success', 'message': 'Files received and processed', 'answer': answer})
    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'}, status=405)

def get_answer(image, text, caption):
    global client

    prompt = f'''[Caption of Figure] {caption}

[Question] {text}'''

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Please answer the question with reference to the image."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": image}
                }
            ]}
        ]
    )

    answer = response.choices[0].message.content
    return answer

def get_answer_llava(img_str, text, caption):
    global device, model, processor

    image = base64.b64decode(img_str.replace('data:image/png;base64,', ''))
    image = Image.open(io.BytesIO(image)).convert("RGB")

#     prompt = f'''[INST] Please answer the question with reference to the image.
# <image>
# Caption of Figure: {caption}
# Question: {text} [/INST]'''

    prompt = f"Please answer the question with reference to the image. USER: <image>\nCaption of Figure: {caption}\nQuestion: {text}\nASSISTANT:"
    
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    response = processor.decode(output[0], skip_special_tokens=True)

    # answer = response.split('[/INST]')[1]
    answer = response.split('ASSISTANT:')[1]

    return answer

