import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import make_grid
import os
from torchgeo.models import ResNet18_Weights,ResNet50_Weights



class MultiStreamAttention(nn.Module):
    def __init__(self, in_channels = 2048, K = 224):
        super(MultiStreamAttention, self).__init__()
        
        # Attention network Φ with 5x5 and 1x1 convolutions
        self.attention_fcn = nn.Sequential(
            nn.Conv2d(in_channels*2, K, kernel_size=5, padding=2),
            nn.BatchNorm2d(K),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, 2, kernel_size=1)
        )
        
    def forward(self,features_list):
        """
        Args:
            features_list: List of feature maps [rgb_features, aux_features]
            Each feature map has shape [batch_size, channels, height, width]
        """
        
        batch_size, channels, height, width = features_list[0].shape
        concat_features = torch.cat(features_list, dim=1)
        
        attention_scores = self.attention_fcn(concat_features)
        attention_weights = torch.sigmoid(attention_scores)  # [B, 2, H', W']
        
        weighted_features = []
        pre_attention_outputs = []
        
        for i, features in enumerate(features_list):
            weights = attention_weights[:, i:i+1, ...]
            weighted_features.append(features * weights)
            pre_attention_outputs.append(features)
        
        # Sum the weighted features
        merged_features = sum(weighted_features)
        
        return merged_features, pre_attention_outputs, attention_weights

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(SelfAttentionModule, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        
    def forward(self, features_list):
        # Concatenate features
        F_concat = torch.cat(features_list, dim=1)  # [B, 4096, H, W]
        batch_size, channels, height, width = F_concat.shape
        
        # Generate Q, K, V
        Q = self.query_conv(F_concat)  # [B, 2048, H, W]
        K = self.key_conv(F_concat)    # [B, 2048, H, W]
        V = self.value_conv(F_concat)  # [B, 4096, H, W]
        
        # Reshape for attention computation
        Q = Q.view(batch_size, -1, height * width)  # [B, 2048, HW]
        K = K.view(batch_size, -1, height * width)  # [B, 2048, HW]
        V = V.view(batch_size, -1, height * width)  # [B, 4096, HW]
        
        # Compute attention scores
        attention = torch.bmm(Q.permute(0, 2, 1), K)  # [B, HW, HW]
        attention = F.softmax(attention / torch.sqrt(torch.tensor(channels)), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(V, attention.permute(0, 2, 1))  # [B, 4096, HW]
        out = out.view(batch_size, -1, height, width)   # [B, 4096, H, W]
        
        # Split back to two streams
        F_RGB_prime = out[:, :2048, :, :]
        F_I_prime = out[:, 2048:, :, :]
        
        # Sum the features
        F_final = F_RGB_prime + F_I_prime
        
        pre_attention_outputs = []
        for i, features in enumerate(features_list):
            pre_attention_outputs.append(features)
        
        return F_final, pre_attention_outputs, attention
    
    
class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(CrossAttentionModule, self).__init__()
        
        # Transformations for RGB stream
        self.query_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Transformations for Indices stream
        self.query_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def attention(self, Q, K, V):
        batch_size, channels, height, width = Q.shape
        
        # Reshape for attention computation
        Q = Q.view(batch_size, -1, height * width)  # [B, C, HW]
        K = K.view(batch_size, -1, height * width)  # [B, C, HW]
        V = V.view(batch_size, -1, height * width)  # [B, C, HW]
        
        # Compute attention scores
        attention = torch.bmm(Q.permute(0, 2, 1), K)  # [B, HW, HW]
        attention = F.softmax(attention / torch.sqrt(torch.tensor(channels)), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(V, attention.permute(0, 2, 1))  # [B, C, HW]
        return out.view(batch_size, channels, height, width)
    
    def forward(self, features_list):
        F_rgb, F_indices = features_list
        
        # RGB attending to Indices
        Q_rgb = self.query_rgb(F_rgb)
        K_ind = self.key_indices(F_indices)
        V_ind = self.value_indices(F_indices)
        F_rgb_attended = self.attention(Q_rgb, K_ind, V_ind)
        
        # Indices attending to RGB
        Q_ind = self.query_indices(F_indices)
        K_rgb = self.key_rgb(F_rgb)
        V_rgb = self.value_rgb(F_rgb)
        F_ind_attended = self.attention(Q_ind, K_rgb, V_rgb)
        
        # Combine attended features
        F_final = F_rgb_attended + F_ind_attended
        
        pre_attention_outputs = []
        for i, features in enumerate(features_list):
            pre_attention_outputs.append(features)
        
        attention_weights = torch.stack([F_rgb_attended, F_ind_attended], dim=1)
        return F_final, pre_attention_outputs, attention_weights    
    
class AttentionSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(AttentionSegmentationModel, self).__init__()
        self.shared_backbone_1 = backbone
        self.shared_backbone_2 = backbone
        # self.share
        self.channel_attention = MultiStreamAttention(in_channels=2048)
        
        # self.channel_attention = CrossAttentionModule(in_channels=2048)
        
        # Feature normalization layers
        self.rgb_norm = nn.BatchNorm2d(2048)
        self.aux_norm = nn.BatchNorm2d(2048)
        
        # Final prediction layers
        self.conv1x1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
        # Stream-specific prediction layers
        self.stream_predictors = nn.ModuleList([
            nn.Conv2d(2048, num_classes, kernel_size=1)
            for _ in range(2)  # One for each stream
        ])
        
    def forward(self, x):
        # print(x.shape)
        # Split input into RGB and auxiliary channels
        rgb_input = x[:, :3, :, :]
        aux_input = x[:, 3:, :, :]
        
        # Get features from both streams using shared backbone
        rgb_features = self.shared_backbone_1(rgb_input)[0]
        aux_features = self.shared_backbone_2(aux_input)[0]
        
                # Normalize features
        rgb_features = self.rgb_norm(rgb_features)
        aux_features = self.aux_norm(aux_features)
        
        # Apply channel attention
        merged_features, pre_attention_features, attention_weights = self.channel_attention(
            [rgb_features, aux_features]
        )
        
        # Generate predictions
        final_pred = self.conv1x1(merged_features)
        final_pred = self.upsample(final_pred)
        
        # Generate pre-attention predictions for each stream
        stream_preds = []
        for i, features in enumerate(pre_attention_features):
            pred = self.stream_predictors[i](features)
            pred = self.upsample(pred)
            stream_preds.append(pred)
            
        return final_pred, stream_preds, attention_weights



    
def modify_input_layer(original_conv, target_in_channels):
    """Modify input convolution layer for different number of input channels."""
    new_conv = nn.Conv2d(
        target_in_channels, 
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    
    # Use pretrained weights for first 3 channels
    new_conv.weight.data[:, :3, :, :] = original_conv.weight.data[:, :3, :, :]
    
    # Initialize additional channels with mean of RGB weights
    if target_in_channels > 3:
        avg_weights = original_conv.weight.data[:, :3, :, :].mean(dim=1, keepdim=True)
        for i in range(3, target_in_channels):
            new_conv.weight.data[:, i:i+1, :, :] = avg_weights
            
    if original_conv.bias is not None:
        new_conv.bias.data = original_conv.bias.data
        
    return new_conv

def create_backbone(in_channels=3, pretrained=True, weights = "sentinel"):
    
    if weights == "landsat":
        weights = ResNet50_Weights.LANDSAT_ETM_SR_MOCO
    elif weights == "sentinel":
        weights = ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS
        
        
    """Create and modify ResNet backbone."""
    backbone = timm.create_model('resnet50', in_chans=in_channels, features_only=True, out_indices=(4,), pretrained=True)
    
    pretrained_state_dict = weights.get_state_dict(progress=True)
    
    # backbone.load_state_dict(pretrained_state_dict, strict=False)
    
    if in_channels != 3:
        backbone.conv1 = modify_input_layer(backbone.conv1, in_channels)
    
    return backbone








import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Tuple, List, Optional, Dict


class PretrainedModel(nn.Module):
    """
    A wrapper class for pretrained segmentation models that returns both encoder
    and decoder outputs.
    
    Args:
        model_name (str): Name of the segmentation model architecture
        in_channels (int): Number of input channels
        classes (int): Number of output classes
        hidden_dim (int, optional): Hidden dimension for intermediate features. Defaults to 16
        encoder_name (str, optional): Name of encoder backbone. Defaults to "resnet34"
        encoder_weights (str, optional): Pre-trained weights for encoder. Defaults to "imagenet"
        encoder_depth (int, optional): Depth of encoder. Defaults to 5
        decoder_attention_type (Optional[str], optional): Type of attention in decoder. Defaults to None
        activation (Optional[str], optional): Final activation function. Defaults to None
    """
    
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        classes: int,
        hidden_dim: int = 16,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        encoder_depth: int = 5,
        decoder_attention_type: Optional[str] = None,
        activation: Optional[str] = None,
        weights: Optional[str] = 'landsat',
        pretrained: Optional[bool] = True
    ):
        super().__init__()
        
        # Create decoder channels tuple based on hidden_dim
        decoder_channels = tuple([hidden_dim * (2 ** i) for i in range(encoder_depth - 1, -1, -1)])
        
        self.model = AttentionSegmentationModel(create_backbone(in_channels=3, pretrained=pretrained, weights = weights), classes)
        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Tuple containing:
                - encoder_output: Final encoder output
                - decoder_outputs: List of intermediate decoder outputs
        """
        # Get encoder features
        # features = self.model.shared_backbone.encoder(x)
        
        final_pred, stream_preds, attention_weights = self.model(x)
        outputs = {
            'encoder_feature': None,#features,
            'final_pred': final_pred,
            'stream_pred': stream_preds,
            'attention_weights': attention_weights,
            'logits': final_pred
        }
            
        return outputs
