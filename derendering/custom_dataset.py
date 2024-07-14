from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict
import cv2
import numpy as np
import torch

class CustomDataset(Dataset):
    """
    Create regular PyTorch Dataset. 
    
    This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt).
    """

    def __init__(
        self,
        dataset_name_or_path: str
    ):
        super().__init__()

        dataset = load_dataset(dataset_name_or_path, split='train')
        dataset = dataset.filter(lambda example: example['query'] == '<extract_data_table>')
        dataset = dataset.filter(lambda example: self.contains_non_ascii(example['label']) == False)
        self.dataset = dataset.select(range(min(10000, len(dataset))))
        
        self.dataset_length = len(self.dataset)
    
    def contains_non_ascii(self, text):
        return any(ord(char) > 127 for char in text)

    def __len__(self) -> int: 
        # return the length of the dataset
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original Receipt image
            target_sequence(text string) : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # inputs
        img_name = sample['imgname']
        img_path = f'../UniChart_Images/{img_name}'
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.astype(np.float32) / 255.0
        # image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        label = sample['label']

        return image, label