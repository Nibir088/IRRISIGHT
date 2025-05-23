import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamVisionModel, SamImageProcessor

class SAMSegmentation(nn.Module):
    def __init__(self, num_classes: int, freeze_model: bool = True):
        super().__init__()
        self.backbone = SamVisionModel.from_pretrained("facebook/sam-vit-base")
        self.processor = SamImageProcessor.from_pretrained("facebook/sam-vit-base")
        self.decoder = nn.Conv2d(256, num_classes, kernel_size=1)
        
        if freeze_model:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x_resized = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
        with torch.no_grad():
            outputs = self.backbone(x_resized)
            features = outputs.last_hidden_state  # [B, 256, 64, 64]

        features = F.interpolate(features, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.decoder(features)
        logits = logits.to(torch.float32)
        # print(logits.shape, logits.dtype)
        return {'logits': logits}


if __name__ == "__main__":
    model = SAMSegmentation(num_classes=2)
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    dummy_input = torch.rand(2, 3, 224, 224).to(next(model.parameters()).device)  # 3-channel RGB in [0,1]
    with torch.no_grad():
        output = model(dummy_input)

    print("Logits shape:", output['logits'].shape)  # Expected: [2, 2, 256, 256]