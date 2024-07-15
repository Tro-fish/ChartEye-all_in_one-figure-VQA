import io
import base64
from .extract import extract_images_from_pdf, extract_images_from_pptx, extract_images_from_docx
from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings

def extract(request):
    if request.method == 'POST':
        files = request.FILES
        
        results = []
        for key, file in files.items():
            print(f'File field: {key}, Filename: {file.name}, Size: {file.size}')
            
            if file.name.lower().endswith('.pdf'):
                images = extract_images_from_pdf(file.read())
            elif file.name.lower().endswith('.ppt') or file.name.lower().endswith('.pptx'):
                images = extract_images_from_pptx(file.read())
            elif file.name.lower().endswith('.doc') or file.name.lower().endswith('.docx'):
                images = extract_images_from_docx(file.read())
            print(len(images))
            
            results.extend(images)

        base64_images = []
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")  # Save the image to a byte buffer
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(img_str)

        return JsonResponse({'status': 'success', 'message': 'Files received and processed', 'images': base64_images})
    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'}, status=405)