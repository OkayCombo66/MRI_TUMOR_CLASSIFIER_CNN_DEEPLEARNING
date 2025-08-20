import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MRIDataset(Dataset):
    def __init__(self, root="Data/brain_tumor_dataset", img_size=128, mode="train", seed=42):
        
        assert mode in {"train", "val", "test"}

        self.img_size = img_size
        self.mode = mode
        
        tumor, healthy = [], []

    def read_and_resize(path):
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.img.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1))
        return img.astype(np.float32)

        for f in glob.iglob(f"{root}/yes/*.jpg"):
            img = read_and_resize(f)
            if img is not None:
                tumor.append(img)

        for f in glob.iglob(f"{root}/no/*.jpg"):
            img = read_and_resize(f)
            if img is not None:
                tumor.append(img)
        
        
        tumor = np.array(self.tumor,dtype=np.float32)
        healthy = np.array(self.healthy,dtype=np.float32)

        tumor_labels = np.ones(self.tumor.shape[0], dtype=np.float32)
        healthy_labels = np.zeros(self.healthy.shape[0], dtype=np.float32)

        images = np.concatenate((self.tumor, self.healthy), axis=0)
        labels = np.concatenate((self.tumor_labels, self.healthy_labels), axis=0)

        images = self.images/255.0 

        
        X_train, X_temp, y_train, y_temp = \
        train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)


        X_val, X_test, y_val, y_test = \
        train_test_split(X_temp, y_temp, test_size=1/3, random_state=seed, stratify=y_temp)

        if mode == "train":
            self.X, self.y = X_train, y_train
        elif mode == "val":
            self.X, self.y = X_val, y_val
        elif mode == "test":
            self.X, self.y = X_test, y_test


    def __len__(self):
        if self.mode == "train":
            return self.X_train.shape[0]
        elif self.mode == "val":
            return self.X_val.shape[0]
        elif self.mode == "test":
            return self.X_test.shape[0]
            
    def __getitem__(self, index):
        if self.mode == "train":
            return {"image": self.X_train[index], "label": self.y_train[index]}
        elif self.mode == "val":
            return {"image": self.X_val[index], "label": self.y_val[index]}
        elif self.mode == "test":
            return {"image": self.X_test[index], "label": self.y_test[index]}