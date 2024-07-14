from utils import train_collate_fn, eval_collate_fn
from torch.utils.data import DataLoader
from nltk import edit_distance
import lightning as L
import numpy as np
import torch
import re

class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, pixel_values, labels = batch

        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             pixel_values=pixel_values,
                             labels=labels)
        loss = outputs.loss

        # Log the training loss and batch size
        self.log("train_loss", loss, batch_size=self.batch_size * self.config.get("accumulate_grad_batches", 1))

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        MAX_LENGTH = 512
        input_ids, attention_mask, pixel_values, answers = batch

        # Autoregressively generate token IDs
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                            pixel_values=pixel_values, max_new_tokens=MAX_LENGTH)
        # Turn them back into text, chopping off the prompt
        # Important: we don't skip special tokens here, because we want to see them in the output
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred).lower()  # 소문자로 변환
            answer = answer.lower()  # 소문자로 변환
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores), prog_bar=True, batch_size=self.batch_size)

        return scores

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=16)