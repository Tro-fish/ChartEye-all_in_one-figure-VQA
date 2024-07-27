<div align="center">

# ChartEye: Multi Modal QA Chatbot for Scientific Figure Images
<br><b> ChartEye: Vision for Your Scientific Chart & Figure </b> <br><br>
!!!!!!!!!여기에 gif 파일 넣기!!!!!!!!! <br> <br>
<p align="center">
<img src="https://github.com/user-attachments/assets/69ba6f4b-9691-46e4-9ef6-cf19cf4e9212" alt="Description of the image" width="100%" />
</p>

This repository contains the code for ChartEye, including its implementation and execution instructions, for the Figure-to-Caption and QA Chatbot for Chemistry and Materials Science documents.
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
6. Extract number, legend from selected figure image with OCR + create a caption<br>
7. User enters a question --> Input Question, Table, Image, and Caption into LLaVA model and generate answer<br>

## Install
### Setup `python` environment
```
conda create -n charteye python==3.10.10
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

## Implementation Details (technical report)
In this section, we go into detail about how we implemented the functionality for each step.<br> If you simply want to use our application, you don't need to read it.
### Step1
