from custom_dataset import CustomDataset
from model import PaliGemmaModelPLModule
from datasets import load_dataset
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from huggingface_hub import HfApi
from peft import get_peft_model, LoraConfig
from nltk import edit_distance
import json
import PIL
import lightning as L
import torch
import re
import numpy as np

REPO_ID = "google/paligemma-3b-pt-224"
FINETUNED_MODEL_ID = "Trofish/paligemma-cord-demo"
MAX_LENGTH = 512
WANDB_PROJECT = "paligemma"
WANDB_NAME = "demo"

# temporary dataset
train_dataset = CustomDataset("naver-clova-ix/cord-v2", split="train")
val_dataset = CustomDataset("naver-clova-ix/cord-v2", split="validation")

"""
train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=2, shuffle=True)
input_ids, token_type_ids, attention_mask, pixel_values, labels = next(iter(train_dataloader))

val_dataloader = DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=2, shuffle=False)
input_ids, attention_mask, pixel_values, answers = next(iter(val_dataloader))
"""

# use this for full fine-tuning
# model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID)

### only train the language model and freeze the vision encoder (SigLIP) and multimodal projector. ###
# for param in model.vision_tower.parameters():
#     param.requires_grad = False

# for param in model.multi_modal_projector.parameters():
#     param.requires_grad = False


# use this for Q-LoRa
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(REPO_ID)
config = {"max_epochs": 10,
          "val_check_interval": 0.5, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 2,
          #"seed":2022,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = PaliGemmaModelPLModule(config, processor, model, train_dataset, val_dataset)
early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=[early_stop_callback],
)

trainer.fit(model_module)