import os
import json
from torch.utils.data import Dataset
from typing import Any, Dict
from PIL import Image
from tqdm import tqdm

class CustomDataset(Dataset):
    """
    Create regular PyTorch Dataset.
    
    This class takes a custom dataset consisting of images and their corresponding captions.
    """

    def __init__(
        self,
        image_folder: str,
        json_file_path: str,
        sort_json_key: bool = True,
    ):
        super().__init__()

        self.image_folder = image_folder
        self.sort_json_key = sort_json_key

        with open(json_file_path, 'r') as f:
            self.dataset = json.load(f)
        
        self.dataset_length = len(self.dataset)
        self.gt_token_sequences = []

        for sample in tqdm(self.dataset, total=self.dataset_length, desc="Processing samples"):
            ground_truth = {"text_sequence": sample["caption"]}
            self.gt_token_sequences.append(
                self.json2token(
                    ground_truth,
                    sort_json_key=self.sort_json_key,
                )
            )

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __len__(self) -> int: 
        # return the length of the dataset
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns:
            image : the original image
            target_sequence(text string) : tokenized ground truth sequence
        """
        sample = self.dataset[idx]

        # Construct image file path from image_id
        image_id = sample["image_id"]
        image_filename = f"{image_id:012d}.png"
        image_path = os.path.join(self.image_folder, image_filename)

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get the target sequence
        target_sequence = self.gt_token_sequences[idx]

        return image, target_sequence