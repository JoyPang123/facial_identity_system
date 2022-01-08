import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import wandb

from tqdm import tqdm

from dataset import make_loader
from config import config
from model import SimCLRModel
from utils import plot_points


class Training:
    def __init__(self, args):
        self.args = args
        _, self.train_loader = make_loader(
            batch_size=args.batch_size, csv_path=args.csv_path,
            img_root=args.img_root, size=args.size
        )
        self.model = SimCLRModel(
            model_type=self.args.model_type,
            pretrained=self.args.pretrained,
            out_dim=self.args.out_dim,
        ).to(self.args.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1
        )
        os.makedirs("weight", exist_ok=True)


    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self,):
        model_config = {
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "learning rate": self.args.lr,
            "temerature": self.args.temperature
        }

        run = wandb.init(
            project="facial_identily",
            resume=False,
            config=model_config,
        )

        self.model.train()

        scaler = GradScaler(enabled=self.args.fp16_precision)

        for epoch in range(self.args.epochs):
            tqdm_iter = tqdm(
                self.train_loader,
                bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]",
            )

            epoch_loss = 0.0
            for images in tqdm_iter:
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                tqdm_iter.set_description(f"Epoch: {epoch + 1}")
                tqdm_iter.set_postfix_str(f"loss={loss.item():^7.3f}")

                epoch_loss += loss.item()

            log = {
                "epoch": epoch + 1,
                "loss": epoch_loss / len(tqdm_iter),
                "Image": plot_points(
                    self.model, img_root=self.args.img_root,
                    device=self.args.device, num_points=1000) 
            }
            wandb.log(log)

            # Save the model every 5 epochs
            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f"weight/model_{epoch + 1}.pt")

            # Warmup for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()
                

if __name__ == "__main__":
    simclr = SimCLR(args=config)
    simclr.train()
