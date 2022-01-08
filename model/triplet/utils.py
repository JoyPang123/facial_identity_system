import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@torch.no_grad()
def plot_points(model, device, csv_path, img_root, num_points=10):
    model.eval()

    df = pd.read_csv(csv_path)

    # Choose points
    plot_image_list = df.sample(num_points)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((140, 140)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Get features from the list
    features_list = []
    for idx in range(len(plot_image_list)):
        img = Image.open(os.path.join(img_root, plot_image_list.iloc[idx, 0]))
        img = img.crop(plot_image_list.iloc[idx, 2:])
        img = base_transform(img).unsqueeze(0).to(device)
        features = features_list.append(model.get_features(img).cpu()[0].tolist())

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
