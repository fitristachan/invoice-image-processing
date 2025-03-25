from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import json
import albumentations as A

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
        image = sample["image"]
        image = self.preprocess_image(image)

        ground_truth_str = sample["ground_truth"]
        try:
            ground_truth = json.loads(ground_truth_str)
        except json.JSONDecodeError:
            ground_truth = {}

        return {"image": image, "label": ground_truth}

    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = cv2.resize(image, (600, 600)) / 255.0
        image = self.augment(image=image)['image']
        return np.expand_dims(image, axis=0).astype(np.float32)