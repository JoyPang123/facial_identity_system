import argparse

import numpy as np

import torch
import torch.nn as nn

import wandb

from tqdm import tqdm

from utils import plot_points

from dataset import make_loader
from model import TripletNet


def train(args):
    model_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning rate": args.lr,
    }
    run = wandb.init(
        project="facial_identity",
        resume=False,
        config=model_config,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(
        batch_size=args.batch_size, img_root=args.img_root,
        csv_path=args.csv_path
    )

    model = TripletNet(
        model_type=args.model_type, pretrained=args.pretrained,
        out_dim=args.out_dim
    )
    model = model.to(device)

    # Set up hyper-parameters
    criterion = nn.TripletMarginLoss(margin=args.margin)
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pair_dis = nn.PairwiseDistance(p=2)

    for epoch in range(args.epochs):

        tqdm_iter = tqdm(
            train_loader,
            bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]"
        )

        for idx, batched_data in enumerate(tqdm_iter):
            model.train()

            # Get data and move to device
            input_anchor = batched_data["anchor"].to(device)
            input_positive = batched_data["positive_image"].to(device)
            input_negative = batched_data["negative_image"].to(device)

            anchor, pos, neg = model(input_anchor, input_positive, input_negative)

            # Compute l2 distance of the model
            pos_dists = pair_dis(anchor, pos)
            neg_dists = pair_dis(anchor, neg)

            all_image = (neg_dists - pos_dists < args.margin).cpu().numpy().flatten()
            valid_triplets = np.where(all_image == 1)

            # Compute loss
            loss = criterion(anchor[valid_triplets], pos[valid_triplets], neg[valid_triplets])

            # Update models
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar
            tqdm_iter.set_description(f"Epoch: {epoch + 1}")
            tqdm_iter.set_postfix_str(f"loss={loss.item():^7.3f} batch={len(valid_triplets[0])}/{args.batch_size}")

            if idx % 100 == 0:
                log = {
                    "loss": loss.item(),
                    "Image": plot_points(
                        model, csv_path=args.csv_path,
                        device=device, img_root=args.img_root,
                        num_points=1000
                    )
                }
                wandb.log(log)

        # Save the weight
        torch.save(model.state_dict(), f"{args.weight}/model_{epoch + 1}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path for the csv file for training data"
    )
    parser.add_argument(
        "--img_root", type=str, required=True,
        help="Root for the training images"
    )
    parser.add_argument(
        "--weight", type=str, required=True,
        help="Place for saving the weight"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--margin", type=float, default=0.2,
        help="Margin for triplet loss"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--model_type", type=str, default="resnet18",
        help="Model used for training"
    )
    parser.add_argument(
        "--pretrained", action="store_true",
        default=False, help="Whether to use pretrained weight"
    )
    parser.add_argument(
        "--out_dim", type=int, default=256,
        help="Output dimension of the output"
    )
    args = parser.parse_args()
    train(args)
