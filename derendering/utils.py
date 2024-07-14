from transformers import AutoProcessor

REPO_ID = "google/paligemma-3b-pt-224"
TEXT_PROMPT = "<extract_data_table>"
LABEL_PROMPT = '<s_answer>'
MAX_LENGTH = 512
processor = AutoProcessor.from_pretrained(REPO_ID)

def train_collate_fn(examples):
  images = [example[0] for example in examples]
  texts = [TEXT_PROMPT for _ in range(len(images))]
  labels = [LABEL_PROMPT + example[1] for example in examples]

  inputs = processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding=True,
                     truncation="only_second", max_length=MAX_LENGTH,
                     tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  token_type_ids = inputs["token_type_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]
  labels = inputs["labels"]

  return input_ids, token_type_ids, attention_mask, pixel_values, labels


def eval_collate_fn(examples):
  images = [example[0] for example in examples]
  texts = [TEXT_PROMPT for _ in range(len(images))]
  answers = [LABEL_PROMPT + example[1] for example in examples]

  inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

  input_ids = inputs["input_ids"]
  attention_mask = inputs["attention_mask"]
  pixel_values = inputs["pixel_values"]

  return input_ids, attention_mask, pixel_values, answers

  