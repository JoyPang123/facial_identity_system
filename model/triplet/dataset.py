import os

import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms


class CelebADataset(Dataset):
    def __init__(self, img_root, csv_path, transform=None):
        self.img_root = img_root
        self.csv = pd.read_csv(csv_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((140, 140)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) if transform is None else transform

    def __getitem__(self, idx):
        # Read in image and get class
        img_name = str(self.csv.iloc[idx, 0])
        img_path = os.path.join(self.img_root, img_name)
        anchor_image = Image.open(img_path)
        anchor_image = anchor_image.crop(self.csv.iloc[idx, 2:])
        label = self.csv.iloc[idx, 1]

        # Positive sampling and remove the redundant images and randomly sample from positive
        positives = self.csv[self.csv.iloc[:, 1] == label]
        positives = positives[positives.iloc[:, 0] != img_name]
        positive_frame = positives.sample(n=1)
        positive_image = Image.open(f"{self.img_root}/{positive_frame.iloc[0, 0]}")
        positive_image = positive_image.crop(positive_frame.iloc[0, 2:])

        # Negative sampling and randomly sample from positive
        negatives = self.csv[self.csv.iloc[:, 1] != label]
        negative_frame = negatives.sample(n=1)
        negative_image = Image.open(f"{self.img_root}/{negative_frame.iloc[0, 0]}")
        negative_image = negative_image.crop(negative_frame.iloc[0, 2:])

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return {
            "anchor": anchor_image,
            "positive_image": positive_image,
            "negative_image": negative_image
        }

    def __len__(self):
        return len(self.csv)


def make_loader(batch_size, img_root, csv_path):
    train_dataset = CelebADataset(
        img_root=img_root, csv_path=csv_path
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True,
    )

    return train_loader
