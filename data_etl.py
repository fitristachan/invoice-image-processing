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

    def extract_receipt_data(self, ground_truth):
        parsed = self.parse_ground_truth(ground_truth)
        menu_items = []
        total_price = 0.0

        # Safely extract menu data
        menu = parsed.get("gt_parse", {}).get("menu", [])
        
        # Handle different menu formats
        if isinstance(menu, list):
            # Case 1: List of proper item dictionaries
            if menu and all(isinstance(item, dict) for item in menu):
                menu_df = pd.DataFrame(menu)
            # Case 2: List of scalar values (unstructured data)
            else:
                menu_df = pd.DataFrame({'raw_items': menu})
        # Case 3: Single dictionary or other format
        elif isinstance(menu, dict):
            menu_df = pd.DataFrame([menu])
        # Case 4: Empty or invalid data
        else:
            menu_df = pd.DataFrame(columns=['item_name', 'quantity', 'price'])
        
        # Process valid menu data
        if not menu_df.empty:
            # Clean column names (remove special characters, lowercase)
            menu_df.columns = menu_df.columns.str.lower().str.replace(r'[^a-z0-9]', '')
            
            # Find relevant columns with fallbacks
            name_col = self._find_item_name_column(menu_df) or 'item_name'
            qty_col = self._find_quantity_column(menu_df) or 'quantity'
            price_col = self._find_price_column(menu_df) or 'price'
            
            # Add missing columns with defaults
            if name_col not in menu_df.columns:
                menu_df[name_col] = ""
            if qty_col not in menu_df.columns:
                menu_df[qty_col] = 1
            if price_col not in menu_df.columns:
                menu_df[price_col] = 0.0
            
            # Convert to standardized format
            for _, item in menu_df.iterrows():
                try:
                    menu_items.append({
                        "item_name": str(item[name_col]),
                        "quantity": float(str(item[qty_col]).strip() or 1),
                        "price": float(str(item[price_col]).replace(",", "").strip() or 0)
                    })
                except (ValueError, AttributeError) as e:
                    print(f"Error processing menu item: {e}")
                    continue

        # Safely extract total price
        try:
            total_str = str(parsed.get("gt_parse", {}).get("total", {}).get("total_price", "0"))
            total_price = float(total_str.replace(",", "").strip())
        except (ValueError, AttributeError) as e:
            print(f"Error processing total price: {e}")
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