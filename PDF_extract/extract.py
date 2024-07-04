import fitz  # PyMuPDF
import os
import cv2
import numpy as np

def extract_images_from_pdf(pdf_path, output_folder):
    # PDF 파일 열기
    document = fitz.open(pdf_path)
    
    # 출력 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 각 페이지를 순회하며 이미지 추출
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        image_list = page.get_images(full=True)
        
        # 페이지 내의 모든 이미지 저장
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]  # 이미지의 xref 번호
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]  # 이미지 데이터
            
            # 이미지 파일로 저장
            image_filename = f"image{page_num + 1}_{img_index}.png"
            image_filepath = os.path.join(output_folder, image_filename)
            with open(image_filepath, "wb") as img_file:
                img_file.write(image_bytes)
            
            print(f"Saved {image_filename}")

    # 문서 닫기
    document.close()

def classify_chart_images(input_folder, output_folder):
    # 출력 폴더 확인 및 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 입력 폴더의 모든 이미지 파일 검사
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            
            # 이미지 읽기
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 엣지 검출을 통한 패턴 분석
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # 차트의 특징적인 선분 검출
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
            if lines is not None:
                # 선분이 충분히 많다면 차트로 분류
                if len(lines) > 10:  # 이 값은 실험적으로 조정해야 할 수 있음
                    chart_image_path = os.path.join(output_folder, filename)
                    cv2.imwrite(chart_image_path, image)
                    print(f"Chart detected and saved: {chart_image_path}")


pdf_path = 'data.pdf'  # PDF 파일 경로
output_folder = 'images'  # 추출된 이미지를 저장
output_folder2 = 'figures' # 추출된 figures를 저장
extract_images_from_pdf(pdf_path, output_folder)
classify_chart_images(output_folder, output_folder2)