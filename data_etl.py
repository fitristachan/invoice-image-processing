from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import json
import albumentations as A
import pandas as pd

class DatasetReceipt:
    def __init__(self, dataset_name="naver-clova-ix/cord-v2", split="train"):
        self.dataset = load_dataset(dataset_name, split=split)
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=10, p=0.5)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.preprocess_image(sample["image"])
        ground_truth = self.parse_ground_truth(sample["ground_truth"])
        
        # Extract menu items and total price
        menu_df, total_price = self.extract_receipt_data(ground_truth)
        
        return {
            "image": image,
            "menu_df": menu_df,      # DataFrame with columns: ['item_name', 'quantity', 'price']
            "total_price": total_price  # Float (e.g., 1591600.0)
        }

    def preprocess_image(self, image):
        """Resize, normalize, and augment image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image = cv2.resize(image, (600, 600)) / 255.0
        image = self.augment(image=image)['image']
        return np.expand_dims(image, axis=0).astype(np.float32)

    def parse_ground_truth(self, ground_truth_str):
        """Parse JSON ground truth string."""
        try:
            return json.loads(ground_truth_str)
        except json.JSONDecodeError:
            return {}

    def extract_receipt_data(self, ground_truth):
        """Convert JSON labels into a DataFrame of menu items and total price."""
        menu_items = []
        
        # Extract menu items if they exist
        if "gt_parse" in ground_truth and "menu" in ground_truth["gt_parse"]:
            for item in ground_truth["gt_parse"]["menu"]:
                menu_items.append({
                    "item_name": item.get("nm", ""),
                    "quantity": item.get("cnt", ""),
                    "price": float(item.get("price", "0").replace(",", ""))
                })
        
        # Extract total price if it exists
        total_price = 0.0
        if "gt_parse" in ground_truth and "total" in ground_truth["gt_parse"]:
            total_price = float(
                ground_truth["gt_parse"]["total"].get("total_price", "0").replace(",", "")
            )
        
        return pd.DataFrame(menu_items), total_price