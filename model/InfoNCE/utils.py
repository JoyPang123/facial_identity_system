import random
import os

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch

import torchvision.transforms as transforms

@torch.no_grad()
def plot_points(model, img_root="img_align_celeba", device="cuda:0", num_points=10):
    model.eval()
    
    # Choose points
    dirs = os.listdir(img_root)
    plot_image_list = random.choices(dirs, k=num_points)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32)
    ])

    # Get features from the list
    features_list = []
    for img_name in plot_image_list:
        img = Image.open(os.path.join(img_root, img_name))
        img = base_transform(img).unsqueeze(0).to(device)
        features = features_list.append(model(img).cpu()[0].tolist())

    # Do the dimension reduction
    two_dim_pca = PCA(n_components=2).fit_transform(features_list)
    two_dim_tsne = TSNE(n_components=2).fit_transform(features_list)

    # Show plot
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("PCA")
    ax1.scatter(two_dim_pca[:, 0], two_dim_pca[:, 1], c=list(range(num_points)))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("TSNE")
    ax2.scatter(two_dim_tsne[:, 0], two_dim_tsne[:, 1], c=list(range(num_points)))
    
    return fig