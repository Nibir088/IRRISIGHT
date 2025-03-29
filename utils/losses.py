import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FocalLoss(nn.Module):
    """
    Focal Loss implementation.
    
    Args:
        gamma (float): Focusing parameter
        alpha (float): Class balance parameter
        
    Attributes:
        gamma (float): Focusing parameter
        alpha (float): Class balance parameter
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            pred (torch.Tensor): Model predictions [B, C, H, W]
            target (torch.Tensor): Target labels [B, H, W]
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        # weights = [0.94, 0.96, 1.04, 52.3]  #---> Utah Best Weight
        # weights = [0.817, 1.938, 0.820, 1.280] #---> WA
        # weights = [0.886, 0.935, 1.076, 18.16] # col
        # weights = [0.044, 0.135, 0.056, 4.00] # FL
        # weights = [1,1,1,5] # multistate
        # weights = [1,1,1,10]
        weights = [1,1,1,1]
        weights_tensor = torch.FloatTensor(weights).cuda()
        ce_loss = F.cross_entropy(pred, target, reduction='none',weight=weights_tensor)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
class DiceLoss(nn.Module):
    """
    Calculate Dice loss with land mask.
    
    Args:
        predictions (torch.Tensor): Model predictions [B, C, H, W]
        targets (torch.Tensor): Target labels [B, H, W]
        land_mask (torch.Tensor): Land use mask [B, H, W]
        smooth (float): Smoothing factor
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Per-class Dice loss
            - Mean Dice loss
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, land_mask: torch.Tensor,) -> torch.Tensor:
        
        num_classes = predictions.shape[1]
        loss_mask = ((land_mask == 1) | (land_mask == 2)).unsqueeze(1).float()
        predictions = predictions * loss_mask
        
        
        targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2) * loss_mask
        
        intersection = torch.sum(predictions * targets_onehot, dim=(2, 3))
        union = torch.sum(predictions, dim=(2, 3)) + torch.sum(targets_onehot, dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
class KGLoss(nn.Module):
    """
    Knowledge-Guided Loss for incorporating domain knowledge using elastic net regularization.
    
    Args:
        alpha (float): Balance parameter between L1 and L2 loss.
                      alpha=1.0 is pure L1, alpha=0.0 is pure L2
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Compute elastic net regularization loss on weights.
        
        Args:
            w (torch.Tensor): Weight tensor to regularize
            
        Returns:
            torch.Tensor: Combined L1 + L2 regularization loss
        """
        # L1 loss
        l1_loss = torch.sum(torch.abs(w))
        
        # L2 loss 
        l2_loss = torch.sum(w ** 2) **0.5
        
        # Combine using alpha parameter
        # alpha controls balance between L1 and L2
        loss = self.alpha * l1_loss + (1 - self.alpha) * l2_loss
        
        return loss
