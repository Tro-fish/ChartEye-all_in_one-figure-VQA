from django.http import JsonResponse
import json
import openai
from dotenv import dotenv_values

config = dotenv_values("../.env")
client = openai.OpenAI(api_key=config.get("API_KEY"))

def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image = data.get('image')
        text = data.get('text')
        caption = data.get('caption')

        answer = get_answer(image, text, caption)
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