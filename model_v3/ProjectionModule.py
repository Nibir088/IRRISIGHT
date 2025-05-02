import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Dict

class ProjectionModule(nn.Module):
    """
    Projects a 1-channel spatial prior using a pretrained ResNet18 and fuses with external logits.

    Args:
        num_classes (int): Number of target output classes.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained resnet18 and modify first layer to accept 1-channel input
        backbone = resnet18(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

        # Learnable weight for fusion
        self.weights = nn.Parameter(torch.ones(1, num_classes, 1, 1))

    def forward(self, logits: torch.Tensor, spatial_prior: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, num_classes, H, W] — model output
            spatial_prior: [B, 1, H, W] — raw 1-channel prior

        Returns:
            Dict[str, Tensor]: {
                'weighted_ensemble': [B, num_classes, H, W]
            }
        """
        # Encode spatial prior using pretrained ResNet
        x = self.encoder(spatial_prior)          # [B, 512, H/32, W/32]
        x = F.interpolate(x, size=logits.shape[-2:], mode='bilinear', align_corners=False)
        prior_projected = self.classifier(x)     # [B, num_classes, H, W]

        # Fuse with logits
        fused = logits + self.weights * prior_projected
        return fused

# ### test the code
# if __name__ == "__main__":
#     # === Configuration ===
#     batch_size = 4
#     num_classes = 5
#     height, width = 224, 224

#     # === Initialize the ProjectionModule ===
#     model = ProjectionModule(num_classes=num_classes)
#     model.eval()  # Switch to eval mode for inference

#     # === Create Dummy Inputs ===
#     logits = torch.randn(batch_size, num_classes, height, width)
#     spatial_prior = torch.randn(batch_size, 1, height, width)

#     # === Forward Pass ===
#     with torch.no_grad():
#         output = model(logits, spatial_prior)

#     # === Print Output Info ===
#     print("✅ Output Shape:", output.shape)
#     print("✅ Output Type:", type(output))
#     print("✅ Output Range:", output.min().item(), "to", output.max().item())
