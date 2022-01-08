import os

import numpy as np

import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as transforms


class GaussianBlur():
    """Blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class CelebADataset(Dataset):
    def __init__(self, csv_path="celebA.csv", 
                 img_root="img_align_celeba", size=224):
        self.img_root = img_root
        self.csv = pd.read_csv(csv_path, index_col=0)
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Read in image
        img_path = os.path.join(self.img_root, str(self.csv.iloc[idx, 0]))
        img = cv2.imread(img_path)

        # Transformation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_one = self.transforms(img)
        img_two = self.transforms(img)

        return img_one, img_two
    
    
def make_loader(batch_size, csv_path, img_root, size):
    dataset = CelebADataset(csv_path=csv_path, img_root=img_root, size=size)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, pin_memory=True
    )

    return dataset, dataloader
