from django.http import JsonResponse
from .inference import captioning
from tqdm import tqdm

def caption(request):
    if request.method == 'POST':
        image_bytes = request.body
        caption = captioning(image_bytes)
        return JsonResponse({'status': 'success', 'message': 'Files received and processed', 'caption': caption})
    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'}, status=405)