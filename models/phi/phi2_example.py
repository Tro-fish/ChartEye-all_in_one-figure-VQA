import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
prompt = "Three letters picked without replacement from qqqkkklkqkkk. Give prob of sequence qql."
inputs = tokenizer(f"Instruct: {prompt} \n\nOutput:", return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=1000)
text = tokenizer.batch_decode(outputs)[0]
print(text)