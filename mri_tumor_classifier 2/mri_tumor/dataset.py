"""Dataset loading for the MRI brain-tumor binary classification task.

Expects a root directory with two subfolders:
    <root>/yes/*.jpg   -> tumor images   (label = 1)
    <root>/no/*.jpg    -> healthy images (label = 0)

This matches the layout of the public "Brain MRI Images for Brain Tumor
Detection" dataset (Navoneel Chakrabarty, Kaggle).
"""

import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, root="data/brain_tumor_dataset", img_size=128, mode="train", seed=42):
        assert mode in {"train", "val", "test"}

        self.root = root
        self.img_size = img_size
        self.mode = mode

        def _read_and_resize(path):
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            return img.astype(np.float32)

        tumor, healthy = [], []

        for f in sorted(glob.iglob(f"{root}/yes/*.jpg")):
            img = _read_and_resize(f)
            if img is not None:
                tumor.append(img)

        for f in sorted(glob.iglob(f"{root}/no/*.jpg")):
            img = _read_and_resize(f)
            if img is not None:
                healthy.append(img)

        if not tumor or not healthy:
            raise RuntimeError(
                f"No images found under '{root}/yes' or '{root}/no'. "
                "Check the `root` path passed to MRIDataset."
            )

        tumor = np.array(tumor, dtype=np.float32)
        healthy = np.array(healthy, dtype=np.float32)

        tumor_labels = np.ones(tumor.shape[0], dtype=np.float32)
        healthy_labels = np.zeros(healthy.shape[0], dtype=np.float32)

        images = np.concatenate((tumor, healthy), axis=0)
        labels = np.concatenate((tumor_labels, healthy_labels), axis=0)

        images = images / 255.0

        # 70% train / 20% val / 10% test, stratified on the label.
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.30, random_state=seed, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 / 3, random_state=seed, stratify=y_temp
        )

        if mode == "train":
            self.X, self.y = X_train, y_train
        elif mode == "val":
            self.X, self.y = X_val, y_val
        else:
            self.X, self.y = X_test, y_test

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return {"image": self.X[index], "label": self.y[index]}
