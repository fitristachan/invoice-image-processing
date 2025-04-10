from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import json
import albumentations as A
import pandas as pd
import re


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
     
    @staticmethod
    def _find_price_column(df):
        """Find price column with priority and fallback logic"""
        # Exact matches first
        exact_matches = ['price', 'unit_price', 'unitprice']
        for col in df.columns:
            if col.lower() in exact_matches:
                return col
        
        # Substring matches with priority
        priority_patterns = [
            'item_net_price',
            'price',
            'unit',
            'amount',
            'value'
        ]
        
        for pattern in priority_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        
        # Fallback to first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else None

    @staticmethod
    def _find_item_name_column(df):
        """Find item name column with comprehensive matching"""
        exact_matches = ['nm', 'name', 'item', 'description', 'menu_item']
        for col in df.columns:
            if col.lower() in exact_matches:
                return col
        
        priority_patterns = [
            'item_desc',
            'item',
            'name',
            'desc',
            'menu',
            'product',
            'title'
        ]
        
        for pattern in priority_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        
        # Fallback to first string column
        string_cols = df.select_dtypes(include=['object']).columns
        return string_cols[0] if len(string_cols) > 0 else None

    @staticmethod
    def _find_quantity_column(df):
        """Find quantity column with smart matching"""
        exact_matches = ['cnt', 'qty', 'quantity', 'count']
        for col in df.columns:
            if col.lower() in exact_matches:
                return col
        
        priority_patterns = [
            'item_qty',
            'qty',
            'quant',
            'cnt',
            'count',
            'amount'
        ]
        
        for pattern in priority_patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        
        # Fallback: look for columns with mostly integer values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if all(x == int(x) for x in unique_vals):
                    return col
        
        return None

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
        parsed = self.parse_ground_truth(ground_truth)
        gt = parsed.get("gt_parse", {})
        menu_items = []
        total_price = "0"

        if "items" in gt:
                items = gt["items"]
                if isinstance(items, list):
                    # Buat DataFrame dari items agar bisa pakai fungsi pencarian kolom
                    items_df = pd.DataFrame(items)
                    if not items_df.empty:
                        items_df.columns = items_df.columns.str.lower()

                        name_col = self._find_item_name_column(items_df) or 'item_desc'
                        qty_col = self._find_quantity_column(items_df) or 'item_qty'
                        price_col = self._find_price_column(items_df) or 'item_net_price'

                        for _, item in items_df.iterrows():
                            try:
                                menu_items.append({
                                    "item_name": str(item.get(name_col, "")),
                                    "quantity": str(item.get(qty_col, "")),
                                    "price": str(item.get(price_col, ""))
                                })
                            except Exception as e:
                                print(f"Skipping malformed item: {item.to_dict()} - Error: {str(e)}")
                                continue

                total_price = gt.get("summary", {}).get("total_net_worth", "0")

        elif "menu" in gt:  # Format CORD
            menu = gt.get("menu", [])
            menu_df = pd.DataFrame(menu) if menu and isinstance(menu, list) else pd.DataFrame()

            if not menu_df.empty:
                menu_df.columns = menu_df.columns.str.lower()

                name_col = self._find_item_name_column(menu_df) or 'item_name'
                qty_col = self._find_quantity_column(menu_df) or 'quantity'
                price_col = self._find_price_column(menu_df) or 'price'

                for _, item in menu_df.iterrows():
                    try:
                        menu_items.append({
                            "item_name": str(item.get(name_col, "")),
                            "quantity": str(item.get(qty_col, "")),
                            "price": str(item.get(price_col, ""))
                        })
                    except Exception as e:
                        print(f"Skipping malformed item: {item.to_dict()} - Error: {str(e)}")
                        continue

            total_price = str(gt.get("total", {}).get("total_price", "0"))

        return pd.DataFrame(menu_items), total_price

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
