from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re, json, requests


with open ("/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/final_validation.json", 'r') as f:
    validation_data = json.load(f)

model_name = "ahmed-masry/unichart-base-960" # pretrained base model
image_path = "/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/images/000007424363.png"

"""
Chart Question Answering --> <chartqa> your question <s_answer>
Open Chart Question Answering --> <opencqa> your question <s_answer>
Chart Summarization --> <summarize_chart> <s_answer>
Data Table Extraction --> <extract_data_table> <s_answer>
"""



input_prompt = "<summarize_chart> <s_answer>"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open(image_path).convert("RGB")
decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=4,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

"""
    do_sample = True,
    top_p=0.8,
    top_k=50
"""

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.split("<s_answer>")[1].strip()
print(sequence)
