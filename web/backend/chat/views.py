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

#gpt4o
def get_answer(figure_path, user_prompt, caption="", ocr_text=""):
    global client 
    with open(figure_path, "rb") as image_file:
        encoded_figure = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages= [
            {
                "role": "user", "content": [
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_figure}",
                        }
                    },
                    {
                        "type": "text", 
                        "text": f"""
                            <Information1>:
                            "If there are multiple charts, please process each chart individually first.

                            Instructions for the whole chart:
                            1. Extract the title of the chart and all text around the chart image. Record the position of each text to refer to it later.
                            2. Check the legend and if there are colors, record what data each color represents. Ensure to accurately record the position and content of the legend to match the colors within the chart.
                                    
                            Instruction for bar chart:
                            1. Identify and record the labels of the x-axis and y-axis. For vertical bar charts, the x-axis is usually at the bottom and the y-axis is on the left. For horizontal bar charts, the x-axis is on the left and the y-axis can be on the right or left.
                            2. If there are annotations on the bars showing text values, record these values. If annotations provide explanations, record the exact position of the bar they refer to.
                            3. If there are no annotations, estimate the values by considering the values and range of the y-axis.

                            Instruction for line chart: 
                            1. Identify and record the labels of the x-axis and y-axis. The x-axis is usually at the bottom, and the y-axis is on the left.
                            2. If there are annotations near the points on the line showing text values, record these values. If annotations provide explanations, record the exact position of the point they refer to.


                            Instruction for pie chart: 
                            1. Identify and record the color of each pie section along with the corresponding text. Ensure to refer to this information later.
                            2. If there are annotations near the sections showing text values, record these values. If annotations provide explanations, record the exact position of the pie section they refer to.
                            
                            Other scatter plots, box plots, etc:
                            Extract data using similar methods as the above charts. Identify the unique features of each chart and extract data accordingly."

                            <Information2>:
                            {caption}
                            {ocr_text}
                            
                            
                            <Question Case>:
                            "case1. If you need to estimate the value for a specific position, carefully consider the range of values on the y-axis or x-axis, the minor tick marks, and how close the value is to other known values to make a reasonable estimation.
                            case2. If the highest or lowest value in the graph is needed, estimate where that value might be located based on the graph's data distribution.
                            case3. For characteristics or information about the graph, extract the text and use it to infer the details."
                            

                            Using the <Information1> and <Information2> methods provided, for the given image
                            "<User Question>:
                            {user_prompt}"
                            
                            Select the appropriate case from the <Question Case> and provide an answer to the <User Question> based on it.
                            
                            <Answer>:
                            
                            """
                    },
                ]
            },
        ]
    )
    return response.choices[0].message.content

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

