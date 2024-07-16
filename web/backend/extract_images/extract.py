import cv2
import numpy as np
import os
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import io
import os
import io
import cv2
import numpy as np
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
from pdf2docx import Converter
from paddleocr import PaddleOCR
import tempfile
import pdfplumber

def is_single_color(image):
    min_val, max_val = image.min(), image.max()
    return max_val - min_val < 10

def is_extreme_aspect_ratio(idx, image, threshold=4):
    height, width, _ = image.shape
    aspect_ratio = max(height / width, width / height)
    return aspect_ratio > threshold

def count_unique_colors(idx, image):
    unique_colors = set(tuple(pixel) for row in image for pixel in row)
    return len(unique_colors)

def extract_images_from_pptx(pptx_bytes):
    def iter_picture_shapes(prs):
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    yield shape
    prs = Presentation(io.BytesIO(pptx_bytes))
    
    output_images = []
    for picture in iter_picture_shapes(prs):
        image = picture.image.blob
        image = Image.open(io.BytesIO(image))
        output_images.append(image)
    
    return output_images

def extract_images_from_docx(docx_bytes):
    doc = Document(io.BytesIO(docx_bytes))
    
    i = 0
    output_images = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image = rel.target_part.blob
            image_array = np.frombuffer(image, np.uint8)
            image_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if is_extreme_aspect_ratio(i, image_array):
                # print(f"Skipped image{i}.jpg 비율 스킵")
                continue
                
            # if count_unique_colors(i, image) < 50:
            #     print(f"image{i}.jpg 컬러 스킵")
            #     continue

            image = Image.open(io.BytesIO(image))
            output_images.append(image)

    return output_images

### version 1
def extract_images_from_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        pdf_to_docx = io.BytesIO()
        cv = Converter(temp_pdf_path)
        cv.convert(pdf_to_docx, start=0, end=None)
        cv.close()
        pdf_to_docx.seek(0)
        return extract_images_from_docx(pdf_to_docx.getvalue())
    finally:
        os.remove(temp_pdf_path)

### version 2
# def extract_images_from_pdf(file):
#     min_width = 0
#     min_height = 0

#     pdf_file = fitz.open(stream=file, filetype="pdf")

#     output_images = []
#     for page_index in range(len(pdf_file)):
#         page = pdf_file[page_index]
#         image_list = page.get_images(full=True)
#         if image_list:
#             print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
#         else:
#             print(f"[!] No images found on page {page_index}")
        
#         for image_index, img in enumerate(image_list, start=1):
#             xref = img[0]
#             base_image = pdf_file.extract_image(xref)
#             image_bytes = base_image["image"]
#             image_ext = base_image["ext"]
#             image = Image.open(io.BytesIO(image_bytes))

#             if image.width >= min_width and image.height >= min_height:
#                 output_images.append(image)
#             else:
#                 print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")
    
#     return output_images

### version 3
# def extract_images_from_pdf(pdf_bytes):
#     images = []
    
#     # Open the PDF from bytes
#     with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
#         for page in pdf.pages:
#             ph = page.height
#             for img in page.images:
#                 # Extract image data
#                 image_box = (img['x0'], ph - img['y1'], img['x1'], ph - img['y0'])
#                 crop = page.within_bbox(image_box)
#                 image = crop.to_image(resolution=400).original
#                 images.append(image)
                    
#     return images

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_images_from_text(figure_path):
    result = ocr.ocr(figure_path, cls=True)
    text_result = []
    # 인식된 텍스트를 출력합니다.
    for line in result:
        for word in line:
            print(word[1][0])
            text_result.append(word[1][0])
    text_result = '\n'.join(text_result)
    return text_result