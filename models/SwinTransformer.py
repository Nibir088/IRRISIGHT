import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class SwinUnet(nn.Module):
    def __init__(self, num_classes, task="segmentation", backbone_name="swin_base_patch4_window7_224", in_channels=3):
        super().__init__()
        
        self.task = task
        self.in_channels = in_channels
        
        if in_channels != 3:
            self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
            
        # Encoder using timm
        self.encoder = timm.create_model(backbone_name, pretrained=True, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()
        
        # Decoder for segmentation task
        self.decoder4 = DecoderBlock(encoder_channels[-1], 512, encoder_channels[-2])
        self.decoder3 = DecoderBlock(512, 256, encoder_channels[-3])
        self.decoder2 = DecoderBlock(256, 128, encoder_channels[-4])
        self.decoder1 = DecoderBlock(128, 64, 0)
        self.decoder0 = DecoderBlock(64, 32, 0)
        
        # Final convolution for segmentation
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

        # Classifier head for classification task
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(encoder_channels[-1], num_classes)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.proj(x)
            
        features = self.encoder(x)
        features = [f.permute(0, 3, 1, 2) for f in features]

        if self.task == "segmentation":
            # Segmentation decoder
            x = features[-1]
            x = self.decoder4(x, features[-2])
            x = self.decoder3(x, features[-3])
            x = self.decoder2(x, features[-4])
            x = self.decoder1(x)
            x = self.decoder0(x)
            return self.final_conv(x)
        
        elif self.task == "classification":
            # Classification head
            x = features[-1]
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

        else:
            raise ValueError(f"Unsupported task: {self.task}")
