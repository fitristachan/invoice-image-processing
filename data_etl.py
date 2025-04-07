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

    def clean_numeric_string(self, value):
        """Robust cleaning of numeric strings with various formats"""
        if pd.isna(value):
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        str_value = str(value).strip()

        # Remove currency symbols (e.g., Rp, $, etc.)
        str_value = re.sub(r"[^\d,.\sx]", "", str_value)

        # Handle multiplication like '210.00 x'
        if 'x' in str_value:
            str_value = str_value.split('x')[0].strip()

        # Replace comma with dot if it's used as decimal
        if ',' in str_value and '.' not in str_value:
            str_value = str_value.replace(',', '.')

        # Remove thousand separators (e.g., 49.000.00 → 49000.00)
        parts = re.split(r'[.,]', str_value)
        if len(parts) > 2:
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
        else:
            cleaned = str_value

        try:
            return float(cleaned)
        except ValueError:
            raise ValueError(f"Failed to parse numeric string: '{value}' → '{cleaned}'")


    def extract_receipt_data(self, ground_truth):
        parsed = self.parse_ground_truth(ground_truth)
        menu_items = []
        total_price = 0.0

        # Process menu data
        menu = parsed.get("gt_parse", {}).get("menu", [])
        menu_df = pd.DataFrame(menu) if menu and isinstance(menu, list) else pd.DataFrame()

        if not menu_df.empty:
            # Standardize column names
            menu_df.columns = menu_df.columns.str.lower()
            
            # Find columns with fallbacks
            name_col = self._find_item_name_column(menu_df) or 'item_name'
            qty_col = self._find_quantity_column(menu_df) or 'quantity'
            price_col = self._find_price_column(menu_df) or 'price'

            # Process each item
            for _, item in menu_df.iterrows():
                try:
                    # Clean numeric values
                    quantity = self.clean_numeric_string(item.get(qty_col, 1))
                    price = self.clean_numeric_string(item.get(price_col, 0))
                    
                    menu_items.append({
                        "item_name": str(item.get(name_col, "")),
                        "quantity": quantity,
                        "price": price
                    })
                except Exception as e:
                    print(f"Skipping malformed item: {item.to_dict()} - Error: {str(e)}")
                    continue

        # Process total price
        try:
            total_str = str(parsed.get("gt_parse", {}).get("total", {}).get("total_price", "0"))
            total_price = self.clean_numeric_string(total_str)
        except Exception as e:
            print(f"Error processing total price: {str(e)}")
            total_price = 0.0

        return pd.DataFrame(menu_items), total_price

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.preprocess_image(sample["image"])
        ground_truth = sample.get("ground_truth", {})
        
        menu_df, total_price = self.extract_receipt_data(ground_truth)
        
        # Convert menu_df to numeric numpy arrays
        try:
            quantities = menu_df['quantity'].astype(float).values
            prices = menu_df['price'].astype(float).values
        except Exception as e:
            print(f"Error converting menu data to numeric: {str(e)}")
            quantities = np.array([], dtype=np.float32)
            prices = np.array([], dtype=np.float32)
        
        return {
            "image": image,
            "quantities": quantities.astype(np.float32),
            "prices": prices.astype(np.float32),
            "total_price": np.float32(total_price),
            "item_names": menu_df['item_name'].values if 'item_name' in menu_df else np.array([])
        }