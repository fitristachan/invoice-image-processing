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

    def parse_ground_truth(self, ground_truth):
        """Safely parse ground truth whether it's a string or dict"""
        if isinstance(ground_truth, str):
            try:
                return json.loads(ground_truth)
            except json.JSONDecodeError:
                return {}
        return ground_truth or {}
    
    def preprocess_image(self, image):
        """Resize, normalize, and augment image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image = cv2.resize(image, (600, 600)) / 255.0
        image = self.augment(image=image)['image']
        return np.expand_dims(image, axis=0).astype(np.float32)

    def extract_receipt_data(self, ground_truth):
        """Robust extraction of menu items and totals"""
        parsed = self.parse_ground_truth(ground_truth)
        menu_items = []
        total_price = 0.0

        # Safely extract menu items
        menu = parsed.get("gt_parse", {}).get("menu", [])
        if isinstance(menu, list):
            for item in menu:
                if not isinstance(item, dict):
                    continue
                try:
                    menu_items.append({
                        "item_name": str(item.get("nm", "")),
                        "quantity": str(item.get("cnt", "")),
                        "price": float(item.get("price", "0"))
                    })
                except (ValueError, AttributeError):
                    continue

        # Safely extract total price
        try:
            total_str = str(parsed.get("gt_parse", {}).get("total", {}).get("total_price", "0"))
            total_price = float(total_str.replace(",", ""))
        except (ValueError, AttributeError):
            total_price = 0.0

        return pd.DataFrame(menu_items), total_price

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.preprocess_image(sample["image"])
        ground_truth = sample.get("ground_truth", {})
        
        menu_df, total_price = self.extract_receipt_data(ground_truth)
        
        return {
            "image": image,
            "menu_df": menu_df,
            "total_price": total_price
        }