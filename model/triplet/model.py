import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, model_type="resnet18", pretrained=True,
                 out_dim=256):
        super().__init__()
        self.model = getattr(models, model_type)(
            pretrained=pretrained
        )
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            out_dim,
            bias=False
        )

    def forward(self, x):
        return self.model(x)


class TripletNet(nn.Module):
    def __init__(self, model_type="resnet18", pretrained=False,
                 out_dim=256):
        super().__init__()
        self.layers = BaseModel(
            model_type=model_type, pretrained=pretrained,
            out_dim=out_dim
        )

    def forward(self, anchor, positive=None, negative=None):
        if self.training:
            anchor_out = F.normalize(self.layers(anchor), p=2, dim=1)
            positive_out = F.normalize(self.layers(positive), p=2, dim=1)
            negative_out = F.normalize(self.layers(negative), p=2, dim=1)
            return anchor_out, positive_out, negative_out
        else:
            anchor_out = F.normalize(self.layers(anchor), p=2, dim=1)
            return anchor_out

    def get_features(self, x):
        return F.normalize(self.layers(x), p=2, dim=1)


class DimFeatures(nn.Module):
    def __init__(self, in_dim=256, out_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.layers(x), p=2, dim=1)
