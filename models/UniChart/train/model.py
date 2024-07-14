from utils import train_collate_fn, eval_collate_fn
from torch.utils.data import DataLoader
from nltk import edit_distance
from torch.nn.utils.rnn import pad_sequence
import lightning as L
import numpy as np
import torch
import re


class UniChartModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch

        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        MAX_LENGTH = 512
        pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )
        outputs = self.model.generate(pixel_values,
                            decoder_input_ids=decoder_prompts,
                            max_length=MAX_LENGTH,
                            early_stopping=True,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=4,
                            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                            return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            pred = pred.split("<s_answer>")[1] 
            pred = pred.replace(self.processor.tokenizer.eos_token, "").replace("<s>", "").strip(' ')
            answer = answer.split("<s_answer>")[1] 
            answer = answer.replace(self.processor.tokenizer.eos_token, "").strip(' ')
            if self.compute_metric(answer, pred):
              scores.append(1)
            else:
              scores.append(0)

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=16)