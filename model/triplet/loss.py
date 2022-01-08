import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        self.pair_dist = nn.PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pair_dist.forward(anchor, positive)
        neg_dist = self.pair_dist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss
