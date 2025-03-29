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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        total_in_channels = in_channels + skip_channels if skip_channels > 0 else in_channels
        self.conv1 = nn.Conv2d(total_in_channels, out_channels, kernel_size=3, padding=1)
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
        self.in_channels = in_channels
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
        
        # print(out.shape)
        # Split back to two streams
        F_RGB_prime = out[:, :self.in_channels, :, :]
        F_I_prime = out[:, self.in_channels:, :, :]
        
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
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))  # Learnable balance factor
        
    def attention(self, Q, K, V):
        batch_size, channels, height, width = Q.shape
        
        # Reshape for attention computation
        Q = Q.view(batch_size, -1, height * width)  # [B, C, HW]
        K = K.view(batch_size, -1, height * width)  # [B, C, HW]
        V = V.view(batch_size, -1, height * width)  # [B, C, HW]
        
        # Compute attention scores
        scale = torch.sqrt(torch.tensor(channels, dtype=torch.float32, device=Q.device))
        attention = torch.bmm(Q.permute(0, 2, 1), K)  # [B, HW, HW]
        attention = F.softmax(attention / scale, dim=-1)
        
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
        F_final = self.fusion_weight * F_rgb_attended + (1 - self.fusion_weight) * F_ind_attended

        
        pre_attention_outputs = []
        for i, features in enumerate(features_list):
            pre_attention_outputs.append(features)
        
        attention_weights = torch.stack([F_rgb_attended, F_ind_attended], dim=1)
        return F_final, pre_attention_outputs, attention_weights
    
# class CrossAttentionModule(nn.Module):
#     def __init__(self, in_channels=2048):
#         super(CrossAttentionModule, self).__init__()
        
#         self.in_channels = in_channels
#         self.scale = (in_channels ** -0.5)  # Scaling factor for attention
        
#         # RGB stream transformations
#         self.query_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.key_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.value_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
#         # Indices stream transformations
#         self.query_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.key_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.value_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

#         # Fusion Layer
#         self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

#     def attention(self, Q, K, V):
#         """
#         Compute scaled dot-product attention.
#         """
#         # Shape: [B, C, H, W] -> [B, C, HW]
#         B, C, H, W = Q.shape
#         Q, K, V = [x.view(B, C, H * W) for x in (Q, K, V)]

#         # Compute attention scores
#         attn = torch.einsum("bcm, bcn -> bmn", Q, K) * self.scale  # [B, HW, HW]
#         attn = F.softmax(attn, dim=-1)

#         # Apply attention to values
#         out = torch.einsum("bmn, bcn -> bcm", attn, V).view(B, C, H, W)  # [B, C, H, W]
#         return out

#     def forward(self, features_list):
#         F_rgb, F_indices = features_list

#         # RGB attends to Indices
#         F_rgb_attended = self.attention(
#             self.query_rgb(F_rgb), self.key_indices(F_indices), self.value_indices(F_indices)
#         )

#         # Indices attend to RGB
#         F_ind_attended = self.attention(
#             self.query_indices(F_indices), self.key_rgb(F_rgb), self.value_rgb(F_rgb)
#         )

#         # Fusion
#         F_final = self.fusion(torch.cat([F_rgb_attended, F_ind_attended], dim=1))

#         return F_final, [F_rgb, F_indices], torch.stack([F_rgb_attended, F_ind_attended], dim=1)
 

    
class AttentionSegmentationModel(nn.Module):
    def __init__(self, backbone, projection, num_classes, backbone_type="swin", attention_type = "self"):
        super(AttentionSegmentationModel, self).__init__()
        self.shared_backbone = backbone  # Only need one backbone since it's shared
        self.proj = projection  # Projection layer if needed
        self.backbone_type = backbone_type
        
        # print(self.backbone_type)
        # self.feat_dim = self.shared_backbone.feature_info.channels()[-1]
        encoder_channels = self.shared_backbone.feature_info.channels()
        self.feat_dim = encoder_channels[-1]
        
        # Decoder path
        self.decoder4 = DecoderBlock(encoder_channels[-1], 512, encoder_channels[-2])
        self.decoder3 = DecoderBlock(512, 256, encoder_channels[-3])
        self.decoder2 = DecoderBlock(256, 128, encoder_channels[-4])
        self.decoder1 = DecoderBlock(128, 64, 0)
        self.decoder0 = DecoderBlock(64, 32, 0)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Attention module
        if attention_type == "stream":
            self.channel_attention = MultiStreamAttention(in_channels=self.feat_dim)
        elif attention_type == "self":
            self.channel_attention = SelfAttentionModule(in_channels=self.feat_dim)
        else:
            self.channel_attention = CrossAttentionModule(in_channels=self.feat_dim)
            
        
        # Feature normalization
        # self.rgb_norm = nn.BatchNorm2d(self.feat_dim)
        # self.aux_norm = nn.BatchNorm2d(self.feat_dim)
        self.rgb_norm = nn.GroupNorm(num_groups=16, num_channels=self.feat_dim)
        self.aux_norm = nn.GroupNorm(num_groups=16, num_channels=self.feat_dim)
           
        self.stream_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.feat_dim, num_classes, kernel_size=1),
                nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
            ) for _ in range(2)
        ])
        
    def forward(self, x):
        # Split input
        rgb_input = x[:, :3, :, :]
        aux_input = x[:, 3:, :, :]
        
        # Project if needed
        if self.proj is not None:
            # rgb_input = self.proj(rgb_input)
            aux_input = self.proj(aux_input)
        
        # Get features
        if self.backbone_type == "swin":
            rgb_features = [f.permute(0,3,1,2) for f in self.shared_backbone(rgb_input)]
            aux_features = [f.permute(0,3,1,2) for f in self.shared_backbone(aux_input)]
        else:
            rgb_features = self.shared_backbone(rgb_input)
            aux_features = self.shared_backbone(aux_input)
        
        # Get deepest features
        rgb_deep = rgb_features[-1] if self.backbone_type == "swin" else rgb_features[0]
        aux_deep = aux_features[-1] if self.backbone_type == "swin" else aux_features[0]
        # print(rgb_deep.shape, self.feat_dim, self.backbone_type)
        # Normalize features
        rgb_deep = self.rgb_norm(rgb_deep)
        # print(rgb_deep.shape)
        aux_deep = self.aux_norm(aux_deep)
        # print(rgb_deep.shape)
        # Apply attention
        merged_features, pre_attention_features, attention_weights = self.channel_attention(
            [rgb_deep, aux_deep]
        )
        
        # Get skip connections
        if self.backbone_type == "swin":
            skip_features = rgb_features[:-1]
        else:
            skip_features = rgb_features[1:]
            skip_features = skip_features[::-1]
        
        # print(merged_features.shape)
        # Decoder pathway
        x = self.decoder4(merged_features, skip_features[-1])
        x = self.decoder3(x, skip_features[-2])
        x = self.decoder2(x, skip_features[-3])
        x = self.decoder1(x)
        x = self.decoder0(x)
        final_pred = self.final_conv(x)
        # print(final_pred.shape)
        # Stream predictions
        stream_preds = []
        for i, (features, predictor) in enumerate(zip(pre_attention_features, self.stream_predictors)):
            stream_preds.append(predictor(features))
        output = {}
        output['rgb_feature'] = rgb_deep
        output['indices_feature'] = aux_deep
        output['msm_final_feature'] = merged_features
        output['msm_features'] = pre_attention_features
        
        output['stream_prediction'] = stream_preds
        output['final_prediction'] = final_pred
        output['stream_attention'] = attention_weights
        
        return output



    
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

def create_backbone(in_channels=15, pretrained=True, weights = "sentinel",backbone_type = "swin"):
    
    if weights == "landsat":
        weights = ResNet50_Weights.LANDSAT_ETM_SR_MOCO
    elif weights == "sentinel":
        weights = ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS
        
    if backbone_type == "swin":
        backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=True, features_only=True)
        if in_channels > 6: ## here 3 for rgb, 3 for indices
            proj = nn.Conv2d(in_channels-3, 3, kernel_size=1)
            return backbone, proj
        return backbone, None
            
    else:    
        """Create and modify ResNet backbone."""
        backbone = timm.create_model('resnet50', in_chans=in_channels, features_only=True, out_indices=(4,), pretrained=True)

        pretrained_state_dict = weights.get_state_dict(progress=True)

        # backbone.load_state_dict(pretrained_state_dict, strict=False)

        if in_channels != 3:
            backbone.conv1 = modify_input_layer(backbone.conv1, in_channels)

    return backbone, None








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
        pretrained: Optional[bool] = True,
        attention_type: Optional[str] = "self"
    ):
        super().__init__()
        
        # Create decoder channels tuple based on hidden_dim
        decoder_channels = tuple([hidden_dim * (2 ** i) for i in range(encoder_depth - 1, -1, -1)])
        
        self.backbone, self.projection = create_backbone(in_channels=in_channels, pretrained=pretrained, weights = weights, backbone_type = model_name)
        self.model = AttentionSegmentationModel(self.backbone, self.projection, classes, attention_type=attention_type, backbone_type = model_name)
        
        
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
        
        output = self.model(x)
        
        
        
        outputs = {
            'encoder_feature': None,#features,
            'rgb_feature': output['rgb_feature'],
            'indices_feature': output['indices_feature'],
            'msm_final_feature':output['msm_final_feature'],
            'msm_features':output['msm_features'],
            'final_pred': output['final_prediction'],
            'stream_pred': output['stream_prediction'],
            'stream_attention': output['stream_attention'],
            'logits': output['final_prediction']
        }
            
        return outputs
