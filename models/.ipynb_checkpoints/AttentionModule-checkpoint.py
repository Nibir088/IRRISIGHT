from typing import Optional
import torch
import torch.nn as nn

class LandUseMask(nn.Module):
    """
    Attention module for land use masking.
    
    Args:
        in_channels (int): Number of input channels
        hidden_dim (int): Number of hidden dimensions
        
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer
        conv2 (nn.Conv2d): Second convolutional layer
        sigmoid (nn.Sigmoid): Sigmoid activation
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mask to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            mask (torch.Tensor): Mask tensor [B, 1, H, W]
            
        Returns:
            torch.Tensor: Masked input tensor [B, C, H, W]
        """
        # print(mask.shape)
        # mask = mask.unsqueeze(1)
        attention_1 = self.conv1(mask)
        attention_2 = self.conv2(attention_1)
        attention_3 = self.sigmoid(attention_2)
        output = {
            'AM-conv1': attention_1,
            'AM-conv2': attention_2,
            'AM-sig': attention_3,
            'attention': attention_3,
            'features': x * attention_3
        }
        return output