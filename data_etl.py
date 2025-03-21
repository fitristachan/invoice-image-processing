from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import json

class DatasetReceipt:
    def __init__(self, dataset_name="naver-clova-ix/cord-v2", split="train"):
        """
        Parameters:
        - dataset_name: Nama dataset dari Hugging Face (default: "naver-clova-ix/cord-v2").
        - split: Split dataset yang akan digunakan (default: "train").
        """
        # Load dataset dari Hugging Face
        self.dataset = load_dataset(dataset_name, split=split)

    def __len__(self):
        """Mengembalikan jumlah sampel dalam dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Mengambil sampel dari dataset berdasarkan indeks.

        Parameters:
        - idx: Indeks sampel yang akan diambil.

        Returns:
        - Dict berisi gambar yang sudah dipreprocess dan label (ground truth).
        """
        sample = self.dataset[idx]
        image = sample["image"]

        # Preprocess the image (menggunakan NumPy dan OpenCV)
        image = self.preprocess_image(image)

        # Get ground_truth dan convert dari JSON string ke dict
        ground_truth_str = sample["ground_truth"]
        print(f"Sample {idx} - Ground Truth String: {ground_truth_str}")  # Debugging

        try:
            ground_truth = json.loads(ground_truth_str)  # Convert to dict
            print(f"Sample {idx} - Parsed Ground Truth: {ground_truth}")  # Debugging
        except json.JSONDecodeError as e:
            print(f"Sample {idx} - Error decoding JSON: {e}")
            ground_truth = {}  # Default jika parsing gagal

        return {"image": image, "label": ground_truth}

    def preprocess_image(self, image):
        """
        Preprocess gambar: resize ke (600, 600) dan convert ke grayscale.

        Parameters:
        - image: Gambar input (PIL.Image atau NumPy array).

        Returns:
        - Gambar yang sudah dipreprocess (NumPy array).
        """
        if isinstance(image, Image.Image):  # Jika formatnya PIL, ubah ke NumPy
            image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize ke (600, 600)
        image = cv2.resize(image, (600, 600))

        # Convert ke grayscale jika masih RGB
        # if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image  # Output dalam format NumPy