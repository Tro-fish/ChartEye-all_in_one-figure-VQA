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

def is_single_color(image):
    min_val, max_val = image.min(), image.max()
    return max_val - min_val < 10

def is_extreme_aspect_ratio(idx, image, threshold=4):
    height, width, _ = image.shape
    aspect_ratio = max(height / width, width / height)
    print(f"idx{idx}: aspect_ratio: {aspect_ratio}")
    return aspect_ratio > threshold

def count_unique_colors(idx, image):
    unique_colors = set(tuple(pixel) for row in image for pixel in row)
    print(f"idx{idx}: unique colors: {len(unique_colors)}")
    return len(unique_colors)

def extract_images_from_pptx(pptx_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    def iter_picture_shapes(prs):
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    yield shape
    prs = Presentation(pptx_path)
    i = 0
    for picture in iter_picture_shapes(prs):
        image = picture.image
        image_bytes = image.blob
        output_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if (not is_single_color(output_image)) and (not is_extreme_aspect_ratio(i, output_image)) and (count_unique_colors(i, output_image) > 50):
            cv2.imwrite(os.path.join(output_folder, f'{i}.jpg'), output_image)
            i += 1

def extract_images_from_docx(docx_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    doc = Document(docx_path)
    i = 0
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image = rel.target_part.blob
            image_array = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if is_extreme_aspect_ratio(i, image):
                print(f"Skipped image{i}.jpg 비율 스킵")
                continue
            # if count_unique_colors(i, image) < 50:
            #     print(f"image{i}.jpg 컬러 스킵")
            #     continue
            image_path = os.path.join(output_folder, f'image{i}.jpg')
            with open(image_path, 'wb') as f:
                f.write(cv2.imencode('.jpg', image)[1].tobytes())
                print(f"Saved {image_path}")
            i += 1

#version 1
# def extract_images_from_pdf(pdf_path, output_folder):
#     # 메모리 내에서 PDF를 DOCX로 변환
#     pdf_to_docx = io.BytesIO()
#     cv = Converter(pdf_path)
#     cv.convert(pdf_to_docx, start=0, end=None)
#     cv.close()
#     pdf_to_docx.seek(0)  # 메모리 파일의 시작점으로 이동 즉, pdf => word => image로 변환하는 거임 
#     extract_images_from_docx(pdf_to_docx, output_folder)

#version 2
def extract_images_from_pdf(pdf_path, output_folder):
    min_width = 0
    min_height = 0
    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # file path you want to extract images from
    file = pdf_path
    # open the file
    pdf_file = fitz.open(file)

    # Iterate over PDF pages
    for page_index in range(len(pdf_file)):
        # Get the page itself
        page = pdf_file[page_index]
        # Get image list
        image_list = page.get_images(full=True)
        # Print the number of images found on this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print(f"[!] No images found on page {page_index}")
        # Iterate over the images on the page
        for image_index, img in enumerate(image_list, start=1):
            # Get the XREF of the image
            xref = img[0]
            # Extract the image bytes
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            # Get the image extension
            image_ext = base_image["ext"]
            # Load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # Check if the image meets the minimum dimensions and save it
            if image.width >= min_width and image.height >= min_height:
                image.save(
                    open(os.path.join(output_folder, f"image{page_index + 1}_{image_index}.png"), "wb"),
                    format='PNG')
            else:
                print(f"[-] Skipping image {image_index} on page {page_index} due to its small size.")

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

print(extract_images_from_text("/Users/hwany/AI-corning/sample_data/2008_images/image0.jpg"))