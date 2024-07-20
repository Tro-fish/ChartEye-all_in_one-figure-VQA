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
processor = LlavaNextProcessor.from_pretrained(model_id)
model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)

def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image = data.get('image')
        text = data.get('text')
        caption = data.get('caption')

        response = get_answer(image, text, caption)
        response = response.split('<Reasoning>:')[-1]
        
        response_split = response.split('<Answer>:')
        reasoning = response_split[0]
        answer = response_split[1]

        return JsonResponse({'status': 'success', 'message': 'Files received and processed', 'reasoning': reasoning, 'answer': answer})
    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'}, status=405)

#gpt4o
def get_answer(image, text, caption="", ocr_text=""):
    global client

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Please give reasoning and answer of the question with reference to the image."},
            {"role": "user", "content": [
                {"type": "text", "text": f"""<Question Case>:
"case1. If you need to estimate the value for a specific position, carefully consider the range of values on the y-axis or x-axis, the minor tick marks, and how close the value is to other known values to make a reasonable estimation.
case2. If the highest or lowest value in the graph is needed, estimate where that value might be located based on the graph's data distribution.
case3. For characteristics or information about the graph, extract the text and use it to infer the details."


Using the <Information1> and <Information2> methods provided, for the given image
"<User Question>:
{text}"

Select the appropriate case from the <Question Case> and provide an answer to the <User Question> based on it.

<Reasoning>:

<Answer>:

"""
                },
                {"type": "image_url", "image_url": {
                    "url": image}
                }
            ]}
        ]
    )

    return response.choices[0].message.content

def get_answer_llava(img_str, text, caption):
    global device, model, processor

    image = base64.b64decode(img_str.replace('data:image/png;base64,', ''))
    image = Image.open(io.BytesIO(image)).convert("RGB")

    prompt = f"Please answer the question with reference to the image. USER: <image>\nCaption of Figure: {caption}\nQuestion: {text}\nASSISTANT:"
    
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    response = processor.decode(output[0], skip_special_tokens=True)
    response = response.split('ASSISTANT:')[1]

    return response

