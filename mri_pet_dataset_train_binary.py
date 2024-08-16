import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data.dataset import Dataset

class MakeDataset(Dataset):
    def __init__(self, image_path):
        self.path = image_path
        file_names = os.listdir(self.path)
        self.data = []
        self.img_labels,  self.mmse_labels = [], []
        for file_name in file_names:
            image_label = int(file_name[0])
            self.img_labels.append(image_label)
            path = os.path.join(self.path, file_name)
            img = sio.loadmat(path)
            image_data = torch.from_numpy(img['data']).float()
            self.data.append(image_data)
        self.data = np.asarray(self.data)
        self.img_labels = np.asarray(self.img_labels)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.img_labels[idx]
