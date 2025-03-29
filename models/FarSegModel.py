import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeo.models import ResNet18_Weights, FarSeg

class FarSegModel(nn.Module):
    def __init__(self, num_classes, backbone_name, in_channels=3):
        super().__init__()
        
        self.in_channels = in_channels
        if in_channels != 3:
            self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
            
        self.model = FarSeg(
            backbone=backbone_name,
            classes=num_classes,
            backbone_pretrained=True
        )

    def forward(self, x):

        if self.in_channels != 3:

            x = self.proj(x)
        
        return self.model(x)