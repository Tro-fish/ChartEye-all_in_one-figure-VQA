from django.http import JsonResponse
import json
import torch
import base64
import io
from PIL import Image
from django.conf import settings
import copy

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

device = settings.DEVICE

pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device, attn_implementation=None)

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

def get_answer(img_str, text, caption="", ocr_text=""):
    global model, tokenizer, device, image_processor

    model.eval()
    model.tie_weights()

    image = base64.b64decode(img_str.replace('data:image/png;base64,', ''))
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "llava_llama_3"

    user_prompt = f"""<Information>:
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

<Question Case>:
"case1. If you need to estimate the value for a specific position, carefully consider the range of values on the y-axis or x-axis, the minor tick marks, and how close the value is to other known values to make a reasonable estimation.
case2. If the highest or lowest value in the graph is needed, estimate where that value might be located based on the graph's data distribution.
case3. For characteristics or information about the graph, extract the text and use it to infer the details."



<User Question>: {text}

You must give response following the template below.

<Reasoning>:

<Answer>:
"""

    user_prompt = DEFAULT_IMAGE_TOKEN + user_prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]


    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=256,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    
    return text_outputs[0]