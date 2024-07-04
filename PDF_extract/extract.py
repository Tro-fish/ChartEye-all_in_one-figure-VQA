import fitz  # PyMuPDF
import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

def extract_images_from_pdf(pdf_path, output_folder):
    min_width = 100
    min_height = 100
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


def classify_chart_images(input_folder, output_folder):
    # 사전 학습된 모델 로드
    model = EfficientNetB0(weights='imagenet')
    
    # 출력 폴더 확인 및 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 입력 폴더의 모든 이미지 파일 검사
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            
            # 이미지 읽기 및 전처리
            image = cv2.imread(file_path)
            image_resized = cv2.resize(image, (224, 224))  # 모델 입력 크기에 맞게 조정
            image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = preprocess_input(image_array)
            
            # 예측 수행
            predictions = model.predict(image_array)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            
            # 차트 이미지로 분류
            for _, label, score in decoded_predictions:
                if 'web_site' in label or 'scoreboard' in label or 'matchstick' in label or 'slide_rule' in label:
                    chart_image_path = os.path.join(output_folder, filename)
                    cv2.imwrite(chart_image_path, image)
                    print(f"Chart detected and saved: {chart_image_path}")
                    continue
            print(f"{filename} has no chart: ", decoded_predictions)

pdf_path = 'data.pdf'  # PDF 파일 경로
output_folder = 'images'  # 추출된 이미지를 저장
output_folder2 = 'figures'  # 추출된 figures를 저장
extract_images_from_pdf(pdf_path, output_folder)
classify_chart_images(output_folder, output_folder2)