<div align="center">

# ChartEye: Multi Modal QA Chatbot for Scientific Figure Images
<br><b> ChartEye: Vision for Your Scientific Chart & Figure </b> <br><br>
!!!!!!!!!여기에 gif 파일 넣기!!!!!!!!! <br> <br>
<p align="center">
<img src="https://github.com/user-attachments/assets/69ba6f4b-9691-46e4-9ef6-cf19cf4e9212" alt="Description of the image" width="100%" />
</p>

This repository contains the code for ChartEye, including its implementation and execution instructions, 
for the Figure-to-Caption and QA Chatbot for Chemistry and Materials Science documents.<br><br>
<b>🏅 Achievement: 2024 [Corning](https://www.corning.com/) AI Challenge Grand Prize 🏅</b>
</div>

## Main features
<b>ChartEye's main features include the following</b>
- Extract Chart(figure) image from PDF, PPTX, WORD files --> *Image Extraction*
- Convert Chart(figure) image to text explanation --> *Chart Captioning*
- Convert imformation in a Chart(figure) image to a number --> *Chart Derendering*
- Answer Question about the information in a Chart(figure) imge --> *Question Answering*


## Overall Pipeline of Chart Eye

<p align="center">
<img src="https://github.com/user-attachments/assets/5adbf683-9807-4e38-bf09-62dbfc9eea8a" alt="Description of the image" width="100%" />
</p>
<b>Step-by-step usage scenarios</b>  <br><br>
1. User uploads PDF, PPT, WORD files<br>
2. Extract images from the input files<br>
3. Extract only Figure images from the images<br>
4. User selects the figure image they want to analyze and answer questions with the chatbot<br>
5. Extract number, legend from selected figure image with OCR + create a caption<br>
6. User enters a question --> Input Question, Table, Image, and Caption into LLaVA model and generate answer<br>

## Install
### Setup `python` environment
```
conda create -n charteye python==3.10.10
conda activate charteye
```
### Install other dependencies
```
pip install -r requirements.txt
```
## Run Server
```
cd frontend
conda activate charteye
npm run dev --> 이거 하고 어디로 들어가야해? localhost? 그것도 적어줘.
```

## (참고) Repository Structure (참고) --> 우리도 이렇게 해야함
The repository has two branches:
- `main` branch contains the code for customizing HuggingFace's implmentation of the UDOP model
- `custom` branch contains the code for customizing Microsoft's implmentation of the UDOP model


``` bash
.
├── LICENSE
├── README.md
├── config/                          # Train/Inference configuration files
│   ├── inference.yaml
│   ├── predict.yaml
│   └── train.yaml
├── core/                            # Main UDOP/DataClass source code
│   ├── common/
│   ├── datasets/
│   ├── models/
│   └── trainers/
├── data/                            # Custom dataset folder
│   ├── images/
│   │   └── image_{idx}.png
│   └── json_data/
│       └── processed_{idx}.pickle
├── main.py                         # Main script to run training/inference
├── models                          # Trained models saved to this folder
├── sweep.py                        # Script to run hyperparameter sweep
├── test                            # Save visualizations during inference
├── requirements.txt
├── udop-unimodel-large-224         # Pretrained UDOP 224 model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
├── udop-unimodel-large-512         # Pretrained UDOP 512 model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── tokenizer_config.json
└── utils                           # Utilities
```

# Implementation Details (technical report)
In this section, we go into detail about how we implemented the functionality for each step.<br> If you simply want to use our application, you don't need to read it.
## Step1 - Image Extraction from files
<p align="center">
<img src="https://github.com/user-attachments/assets/a769a2df-9e95-4930-af72-39ee30d5b572" alt="Description of the image" width="100%" />
<b>STEP1 - Scenario:</b> User uploads PDF, PPT, WORD files & Extract images from the input files<br>
</p>

- Extract images using external tools
    - PDF: Extracting images with [pymupdf](https://pymupdf.readthedocs.io/en/latest/)
    - WORD: Extracting images with [python-docx](https://python-docx.readthedocs.io/en/latest/)
    - PPTX: Extracting images with [python-pptx](https://python-pptx.readthedocs.io/en/latest/)
 
## Step2 - Figure Classification from image
<p align="center">
<img src="https://github.com/user-attachments/assets/632b3fff-a7d8-4cd1-b421-8ea968b82207" alt="Description of the image" width="100%" />
<img src="https://github.com/user-attachments/assets/e16b0251-d9cd-4cb6-9cc8-73c086ac22e8" alt="Description of the image" width="100%" />
<b>STEP2 - Scenario: </b>Extract only Figure images from the images & User selects the figure image they want to analyze and answer questions with the chatbot<br>
</p>

- Using EfficientNet-B4 as a baseline model to train a figure classification model
- Pretrained EfficientNet-B4 model does not have classes for figure, table, so build dataset and proceed with further training
- Our Figure classficiation model can be found [here](https://huggingface.co/hwan99/corning-figure-classifier)

<p align="center">
<img src="https://github.com/user-attachments/assets/36631d3d-1f5c-4da1-b98d-7dd594d1d266" alt="Description of the image" width="100%" />
<b>STEP2 - Figure </b>Classification dataset<br>
</p>

## Step3 - Figure Captioning & Derendering
<p align="center">
<img src="https://github.com/user-attachments/assets/8ca07280-8c27-4389-b09b-6a1114444668" alt="Description of the image" width="100%" />
<b>STEP3 - Scenario: </b>Extract number, legend from selected figure image with OCR + create a caption<br>
</p>

### Step 3-1: Figure Captioning
- Using Paligemma-3B-448 as a baseline model for Figure Captioning models
- Train a text decoder for the Paligemma-3B-448 model with next-word-prediction using [NLMChem](https://www.nature.com/articles/s41597-021-00875-1) dataset
     - Dataset description: Consists of the full text of 150 articles from 67 different chemical journals
     - Enhance domain knowledge of chemistry and materials engineering in text decoder models
     - Number of tokens in the entire NLMChem dataset: 1.2M
     - Train a total of 1K steps with batch size 32 and learning rate 1e-4
- Learn the text decoder first, and then learn both the image encoder and text decoder in the following way
- Train on <figureimage - caption> pair data from domains related to chemistry and materials engineering with [SCICAP](https://huggingface.co/datasets/CrowdAILab/scicap), [SFCC](https://huggingface.co/datasets/mawadalla/scientific-figures-captions-context) datasets
- Train details
  <img width="1000" alt="image" src="https://github.com/user-attachments/assets/eed0cb84-f2ee-4d91-bd95-137cbc625ea8">
- Evaluation details
  <img width="1000" alt="image" src="https://github.com/user-attachments/assets/5f6bb164-f421-4ab3-8648-31fe2a341264">
  <img width="1000" alt="image" src="https://github.com/user-attachments/assets/f1da98ad-127a-4234-a769-96a4a538d315">
  <img width="1000" alt="image" src="https://github.com/user-attachments/assets/8e29c143-71ee-40e0-9091-545589ec0ae4">

### Step 3-2: Figure Derendering
<p align="center">
<img src="https://github.com/user-attachments/assets/e256482f-d99d-470d-ae8c-ca0aa32e6b24" alt="Description of the image" width="100%" />
<b>STEP3-2: Derendering: </b>Using [Paddle OCR](https://github.com/PaddlePaddle/PaddleOCR) for Figure Derendering<br>
</p>

## Step4 - Question Answering
<p align="center">
<img src="https://github.com/user-attachments/assets/922ebbac-3f3e-482a-8a8c-8b0c3edbf8c3" alt="Description of the image" width="100%" />
<b>STEP3 - Scenario: </b>User enters a question --> Input Question, Table, Image, and Caption into LLaVA model and generate answer<br>
</p>

- Uses the LLaVAA-NeXT (LLaMA3-8B base) model, which is currently the most powerful OpenSource VLM model available.
- Input the user's question, along with the <b>figure image, derendered figure, and figure caption extracted in the previous steps,</b> and perform inference to generate an answer.

  <img width="1165" alt="image" src="https://github.com/user-attachments/assets/d455bcb2-420d-4913-9861-1688a362932d">

