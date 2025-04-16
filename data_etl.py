from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import json
import albumentations as A
import pandas as pd
import re
from typing import Dict, Any, Tuple

class DatasetReceipt:
    def __init__(self, dataset_name: str, split: str = "train", image_size: Tuple[int, int] = (600, 600)):
        self.dataset = load_dataset(dataset_name, split=split)
        self.image_size = image_size
        self.dataset_name = dataset_name

        if split == "train":
            self.augment = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.RandomBrightnessContrast(p=0.2),
            ])
        else:
            self.augment = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
            ])
    
    def parse_ground_truth(self, ground_truth: Any) -> Dict[str, Any]:
        if isinstance(ground_truth, str):
            try:
                return json.loads(ground_truth)
            except json.JSONDecodeError:
                return {}
        return ground_truth or {}
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array and apply augmentations"""
        image_np = np.array(image.convert("RGB"))  # PIL Image to numpy (HWC)
        augmented = self.augment(image=image_np)
        augmented_image = augmented["image"]
        
        # Jika formatnya CHW (3, 600, 600), transpose ke HWC (600, 600, 3)
        if augmented_image.shape[0] == 3:  # Cek apakah channel di depan
            augmented_image = np.transpose(augmented_image, (1, 2, 0))  # CHW → HWC
        
        return augmented_image  # Pastikan shape (600, 600, 3)
    
    def extract_cord_data(self, gt: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        menu_items = []
        total_price = "0"
        items = gt.get("items", gt.get("menu", []))
        if isinstance(items, list):
            for item in items:
                menu_items.append({
                    "item_name": str(item.get("item_name", item.get("item_desc", ""))),
                    "quantity": str(item.get("quantity", item.get("item_qty", "1"))),
                    "price": str(item.get("price", item.get("item_net_price", "0")))
                })
        total_price = str(gt.get("summary", {}).get("total_net_worth", 
                         gt.get("total", {}).get("total_price", "0")))
        return pd.DataFrame(menu_items), total_price
    
    def extract_donut_data(self, gt: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        menu_items = []
        total_price = "0"
        gt_parse = gt.get("gt_parse", {})
        if "items" in gt_parse:
            for item in gt_parse["items"]:
                menu_items.append({
                    "item_name": str(item.get("item_desc", "")),
                    "quantity": str(item.get("item_qty", "1")),
                    "price": str(item.get("item_net_price", "0"))
                })
        total_price = str(gt_parse.get("summary", {}).get("total_net_worth", "0"))
        return pd.DataFrame(menu_items), total_price
    
    def extract_receipt_data(self, ground_truth: Any) -> Tuple[pd.DataFrame, str]:
        parsed = self.parse_ground_truth(ground_truth)
        if "gt_parse" in parsed and isinstance(parsed["gt_parse"], dict):
            return self.extract_donut_data(parsed)
        else:
            return self.extract_cord_data(parsed)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        image = sample.get("image", sample.get("img"))
        if image is None:
            raise KeyError(f"Neither 'image' nor 'img' found in sample {idx}")

        image_tensor = self.preprocess_image(image)  # shape: (600, 600, 3)
        menu_df, total_price = self.extract_receipt_data(sample.get("ground_truth", {}))

        return {
            "image": image_tensor,
            "quantities": menu_df['quantity'].values if not menu_df.empty else np.array([]),
            "prices": menu_df['price'].values if not menu_df.empty else np.array([]),
            "total_price": total_price,
            "item_names": menu_df['item_name'].values if not menu_df.empty else np.array([]),
            "dataset_source": self.dataset_name
        }
