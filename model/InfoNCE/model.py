import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class SimCLRModel(nn.Module):
    def __init__(self, model_type="resnet18", 
                 pretrained=False, out_dim=128):
        super().__init__()

        # Create model
        if pretrained == False:
            self.model = getattr(models, "resnet18")(
                num_classes=out_dim
            )
        else:
            self.model = getattr(models, "resnet18")(
                pretrained=True
            )

        dim_mlp = self.model.fc.in_features

        # Add projection head
        self.model.fc = nn.Sequential(
            nn.Linear(dim_mlp, out_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)

