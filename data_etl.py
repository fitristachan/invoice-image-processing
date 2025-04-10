from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import json
import albumentations as A
import pandas as pd
import re

class DatasetReceipt:
    def __init__(self, dataset):
        self.dataset = dataset
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=10, p=0.5)
        ])
        
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

    def determine_sample_type(self, sample):
        """Determine if sample is from CORD or Donut based on its structure"""
        parsed = self.parse_ground_truth(sample.get("ground_truth", {}))
        gt = parsed.get("gt_parse") or parsed.get("ground_truth") or parsed
        
        # Cek struktur CORD
        if "items" in gt or "menu" in gt:
            return "cord"
        # Cek struktur Donut
        elif "line_items" in gt:
            return "donut"
        # Default ke CORD jika tidak bisa ditentukan
        return "cord"

    def extract_receipt_data(self, ground_truth):
        parsed = self.parse_ground_truth(ground_truth)
        gt = parsed.get("gt_parse") or parsed.get("ground_truth") or parsed
        menu_items = []
        total_price = "0"

        sample_type = self.determine_sample_type({"ground_truth": ground_truth})

        if sample_type == "cord":
            if "items" in gt:
                items = gt["items"]
                if isinstance(items, list):
                    for item in items:
                        try:
                            menu_items.append({
                                "item_name": str(item.get("item_name", item.get("item_desc", ""))),
                                "quantity": str(item.get("quantity", item.get("item_qty", ""))),
                                "price": str(item.get("price", item.get("item_net_price", "")))
                            })
                        except Exception as e:
                            print(f"Skipping malformed item: {item} - Error: {str(e)}")
                            continue

                total_price = str(gt.get("summary", {}).get("total_net_worth", "0"))

            elif "menu" in gt:
                menu = gt.get("menu", [])
                if isinstance(menu, list):
                    for item in menu:
                        try:
                            menu_items.append({
                                "item_name": str(item.get("item_name", item.get("item_desc", ""))),
                                "quantity": str(item.get("quantity", item.get("item_qty", ""))),
                                "price": str(item.get("price", item.get("item_net_price", "")))
                            })
                        except Exception as e:
                            print(f"Skipping malformed item: {item} - Error: {str(e)}")
                            continue

                total_price = str(gt.get("total", {}).get("total_price", "0"))
        
        else:  # donut
            if "gt_parse" in parsed:
                gt_parse = parsed["gt_parse"]
                if "line_items" in gt_parse:
                    line_items = gt_parse["line_items"]
                    if isinstance(line_items, list):
                        for item in line_items:
                            try:
                                menu_items.append({
                                    "item_name": str(item.get("item_desc", item.get("description", ""))),
                                    "quantity": str(item.get("item_qty", item.get("quantity", ""))),
                                    "price": str(item.get("item_net_price", item.get("amount", "")))
                                })
                            except Exception as e:
                                print(f"Skipping malformed item: {item} - Error: {str(e)}")
                                continue
                
                total_price = str(gt_parse.get("total", "0"))

        return pd.DataFrame(menu_items), total_price

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.preprocess_image(sample["image"])
        ground_truth = sample.get("ground_truth", {})
        
        menu_df, total_price = self.extract_receipt_data(ground_truth)
        
        return {
            "image": image,
            "quantities": menu_df['quantity'].values if 'quantity' in menu_df else np.array([]),
            "prices": menu_df['price'].values if 'price' in menu_df else np.array([]),
            "total_price": total_price,
            "item_names": menu_df['item_name'].values if 'item_name' in menu_df else np.array([])
        }