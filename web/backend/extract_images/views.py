import io
import base64
from .extract import extract_images_from_pdf1, extract_images_from_pptx, extract_images_from_docx
from .classify import classify_chart_images
from django.http import JsonResponse
from tqdm import tqdm

def extract(request):
    if request.method == 'POST':
        files = request.FILES
        
        # 추출
        extracted_images = []
        for file in tqdm(files.values(), desc='Extracting...', total=len(files)):
            if file.name.lower().endswith('.pdf'):
                images = extract_images_from_pdf1(file.read())
            elif file.name.lower().endswith('.ppt') or file.name.lower().endswith('.pptx'):
                images = extract_images_from_pptx(file.read())
            elif file.name.lower().endswith('.doc') or file.name.lower().endswith('.docx'):
                images = extract_images_from_docx(file.read())
            extracted_images.extend(images)

        # 분류
        classified_images = classify_chart_images(extracted_images)

        # 저장
        base64_images = []
        for image in tqdm(classified_images, desc='Saving...'):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_images.append(img_str)

        # 전달
        return JsonResponse({'status': 'success', 'message': 'Files received and processed', 'images': base64_images})
    return JsonResponse({'status': 'fail', 'message': 'Invalid request method'}, status=405)