from custom_dataset import CustomDataset
from model import UniChartModelPLModule
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from peft import get_peft_model, LoraConfig
from nltk import edit_distance
import json
import PIL
import lightning as L
import torch
import re
import numpy as np

# Set the matmul precision for Tensor Cores
torch.set_float32_matmul_precision('high')

REPO_ID = "ahmed-masry/unichart-chartqa-960"
FINETUNED_MODEL_ID = "Trofish/paligemma-cord-demo"
MAX_LENGTH = 512
WANDB_PROJECT = "paligemma"
WANDB_NAME = "demo"

# 사용 예시
train_images = '/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/train/images'
train_annotations = '/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/train/final_train.json'
val_images = '/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/images'
val_annotations = '/home/wani/Desktop/Corning_team3/dataset/scicap_data/dataset/val/final_validation_v2.json'

train_dataset = CustomDataset(image_folder=train_images, json_file_path=train_annotations)
val_dataset = CustomDataset(image_folder=val_images, json_file_path=val_annotations)

model = VisionEncoderDecoderModel.from_pretrained(REPO_ID,device_map={"": 0})

processor = DonutProcessor.from_pretrained(REPO_ID)

config = {"max_epochs": 5,
          "val_check_interval": 0.3, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 32,
          "lr": 1e-4,
          "batch_size": 1,
          "seed": 2024,
          "num_nodes": 1,
          "warmup_steps": 1000,
          "result_path": "./result",
          "verbose": True,
}

model_module = UniChartModelPLModule(config, processor, model, train_dataset, val_dataset)
early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
#wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)

checkpoint_callback = ModelCheckpoint(
    monitor="val_edit_distance",
    dirpath="./checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    mode="min",
    verbose=True,
)

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
        #logger=wandb_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
)

trainer.fit(model_module)