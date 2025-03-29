from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionModule(nn.Module):
    """
    Combines network logits with spatial priors using learnable weights and projects high-dim spatial priors to class space.
    
    Args:
        num_classes (int): Number of output classes
        in_channels (int): Input channels of spatial prior (e.g., 21)
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 21
    ):
        super().__init__()
        
        # Simple encoder to map spatial priors from in_channels -> num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        
        # Learnable weights for combining with logits
        self.weights = nn.Parameter(torch.full((1, num_classes, 1, 1), 1.0))

    def forward(
        self,
        logits: torch.Tensor,
        spatial_priors_raw: torch.Tensor  # shape: [B, 21, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Combine network logits with spatial priors.

        Args:
            logits: [B, num_classes, H, W]
            spatial_priors_raw: [B, in_channels, H, W]

        Returns:
            Dict[str, torch.Tensor]: Output with combined logits and spatial priors
        """
        # Encode spatial priors to match class dimensions
        spatial_priors = self.encoder(spatial_priors_raw)  # [B, num_classes, H, W]

        if logits.shape != spatial_priors.shape:
            raise ValueError(f"Shape mismatch: logits {logits.shape} != spatial_priors {spatial_priors.shape}")
        
        if logits.shape[1] != self.weights.shape[1]:
            raise ValueError(f"Class mismatch: logits {logits.shape[1]} != weights {self.weights.shape[1]}")
        
        # Weighted ensemble
        ensemble = logits + self.weights * spatial_priors

        return {
            'weighted_ensemble': ensemble,
            'CPM_feature': spatial_priors
        }
