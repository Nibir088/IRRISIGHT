import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def create_backbone(backbone_type="swin"):
    """
    Initializes a feature extraction backbone using the TIMM model zoo.

    Args:
        pretrained (bool): Whether to load pretrained ImageNet weights.
        weights (str): Placeholder for specific pretrained weights (e.g., 'landsat', 'sentinel').
        backbone_type (str): The name of the backbone architecture to use (currently only supports 'swin').

    Returns:
        nn.Module: A feature extractor model that outputs intermediate features.

    Notes:
        - The function is currently configured to use the Swin Transformer base model
          with patch size 4 and window size 7 on 224x224 resolution inputs.
        - The 'features_only=True' flag ensures intermediate layers are exposed
          for decoder use (skip connections).
        - Other backbones can be added later by extending the 'else' clause.
    """
    if backbone_type == "swin":
        backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, features_only=True)
        return backbone

    # Placeholder: Add support for other backbones (e.g., resnet50) if needed
    # Example:
    # backbone = timm.create_model(
    #     'resnet50', in_chans=in_channels,
    #     features_only=True, out_indices=(4,), pretrained=pretrained
    # )

    return backbone



# ========== Attention Modules ==========
class MultiStreamAttention(nn.Module):
    """
    Applies parallel attention over multiple feature maps (typically RGB and auxiliary).

    Args:
        in_channels (int): Number of channels in each feature map.
        K (int): Intermediate number of channels used in attention computation.

    Inputs:
        features_list (List[Tensor]): Two tensors of shape (B, C, H, W).

    Returns:
        Tuple:
            merged_features (Tensor): Weighted sum of input features.
            features_list (List[Tensor]): Original input features.
            attention_weights (Tensor): Attention weights of shape (B, 2, H, W).
    """
    def __init__(self, in_channels=2048, K=224):
        super().__init__()
        self.attention_fcn = nn.Sequential(
            nn.Conv2d(in_channels * 2, K, kernel_size=5, padding=2),
            nn.BatchNorm2d(K),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, 2, kernel_size=1)
        )

    def forward(self, features_list):
        concat_features = torch.cat(features_list, dim=1)
        attention_scores = self.attention_fcn(concat_features)
        attention_weights = torch.sigmoid(attention_scores)
        weighted_features = [features * attention_weights[:, i:i+1] for i, features in enumerate(features_list)]
        merged_features = sum(weighted_features)
        return merged_features, features_list, attention_weights


class SelfAttentionModule(nn.Module):
    """
    Computes self-attention over concatenated RGB and auxiliary feature maps.

    Args:
        in_channels (int): Number of channels in each input feature.

    Inputs:
        features_list (List[Tensor]): Two tensors of shape (B, C, H, W).

    Returns:
        Tuple:
            combined (Tensor): Self-attended merged features of shape (B, C, H, W).
            features_list (List[Tensor]): Original inputs.
            attention (Tensor): Attention matrix of shape (B, HW, HW).
    """
    def __init__(self, in_channels=2048):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, features_list):
        F_concat = torch.cat(features_list, dim=1)
        B, C, H, W = F_concat.shape
        Q = self.query_conv(F_concat).view(B, -1, H * W)
        K = self.key_conv(F_concat).view(B, -1, H * W)
        V = self.value_conv(F_concat).view(B, -1, H * W)
        attention = torch.bmm(Q.permute(0, 2, 1), K)
        attention = F.softmax(attention / torch.sqrt(torch.tensor(C, dtype=torch.float32)), dim=-1)
        out = torch.bmm(V, attention.permute(0, 2, 1)).view(B, -1, H, W)
        F_RGB_prime, F_I_prime = out[:, :self.in_channels], out[:, self.in_channels:]
        return F_RGB_prime + F_I_prime, features_list, attention


class CrossAttentionModule(nn.Module):
    """
    Computes cross-attention between RGB and auxiliary features.

    Args:
        in_channels (int): Number of channels in each input feature.

    Inputs:
        features_list (List[Tensor]): Two tensors (RGB, auxiliary) each of shape (B, C, H, W).

    Returns:
        Tuple:
            fused (Tensor): Fused features using bidirectional cross-attention.
            features_list (List[Tensor]): Original inputs.
            attention_maps (Tensor): Stack of RGB→aux and aux→RGB attended outputs.
    """
    def __init__(self, in_channels=2048):
        super().__init__()
        self.query_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.query_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fusion_weight = nn.Parameter(torch.tensor(0.8))

    def attention(self, Q, K, V):
        B, C, H, W = Q.shape
        Q, K, V = Q.view(B, C, -1), K.view(B, C, -1), V.view(B, C, -1)
        attn = torch.bmm(Q.permute(0, 2, 1), K)
        attn = F.softmax(attn / torch.sqrt(torch.tensor(C, dtype=torch.float32, device=Q.device)), dim=-1)
        return torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)

    def forward(self, features_list):
        F_rgb, F_indices = features_list
        rgb_att = self.attention(self.query_rgb(F_rgb), self.key_indices(F_indices), self.value_indices(F_indices))
        ind_att = self.attention(self.query_indices(F_indices), self.key_rgb(F_rgb), self.value_rgb(F_rgb))
        fused = self.fusion_weight * rgb_att + (1 - self.fusion_weight) * ind_att
        return fused, features_list, torch.stack([rgb_att, ind_att], dim=1)


class DecoderBlock(nn.Module):
    """
    Upsampling decoder block with optional skip connection.

    Args:
        in_channels (int): Input feature channels.
        out_channels (int): Output feature channels.
        skip_channels (int): Number of channels in skip connection.

    Inputs:
        x (Tensor): Input features.
        skip (Tensor or None): Optional skip connection features.

    Returns:
        x (Tensor): Output features after upsampling and convolutions.
    """
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super().__init__()
        total_in_channels = in_channels + skip_channels
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


class SegmentationDecoder(nn.Module):
    """
    Segmentation decoder that stacks multiple DecoderBlocks.

    Args:
        encoder_channels (List[int]): List of encoder feature dimensions.
        out_classes (int): Number of output segmentation classes.

    Inputs:
        x (Tensor): Deepest feature from encoder.
        skips (List[Tensor]): List of skip connection tensors.

    Returns:
        Tensor: Output segmentation map of shape (B, out_classes, H, W).
    """
    def __init__(self, encoder_channels, out_classes):
        super().__init__()
        self.decoder4 = DecoderBlock(encoder_channels[-1], 512, encoder_channels[-2])
        self.decoder3 = DecoderBlock(512, 256, encoder_channels[-3])
        self.decoder2 = DecoderBlock(256, 128, encoder_channels[-4])
        self.decoder1 = DecoderBlock(128, 64)
        self.decoder0 = DecoderBlock(64, 32)
        self.final_conv = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x, skips):
        x = self.decoder4(x, skips[-1])
        x = self.decoder3(x, skips[-2])
        x = self.decoder2(x, skips[-3])
        x = self.decoder1(x)
        x = self.decoder0(x)
        return self.final_conv(x)


class ClassificationDecoder(nn.Module):
    """
    Classification decoder with global average pooling and a linear layer.

    Args:
        feat_dim (int): Dimensionality of input feature.
        out_classes (int): Number of target classes.

    Inputs:
        x (Tensor): Feature tensor of shape (B, C, H, W).

    Returns:
        Tensor: Logits of shape (B, out_classes).
    """
    def __init__(self, feat_dim, out_classes):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, out_classes)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        return self.classifier(pooled)



# ========== Attention Backbone Wrapper ==========
class AttentionBackbone(nn.Module):
    """
    Wrapper for backbone + attention module. 
    Applies an attention mechanism (none, stream, self, cross) on top of backbone features.

    Args:
        backbone (nn.Module): Pretrained CNN or transformer for feature extraction.
        attention_type (str): Type of attention to apply ('none', 'stream', 'self', 'cross').
        in_channels (int): Number of input channels.
        backbone_name (str): Identifier for the backbone type (e.g., 'swin').

    Inputs:
        x (Tensor): Input tensor of shape (B, C, H, W) where C = input channels.

    Returns:
        Tuple:
            merged (Tensor): Output features after attention (B, C, H', W').
            skips (List[Tensor]): List of skip connection features for decoder.
            attn_weights (Tensor or None): Attention maps or None if not used.
    """
    def __init__(self, backbone, attention_type, in_channels, backbone_name):
        super().__init__()
        self.backbone = backbone
        self.attention_type = attention_type
        self.in_channels = in_channels
        self.backbone_type = backbone_name
        self.encoder_channels = self.backbone.feature_info.channels()
        self.feat_dim = self.encoder_channels[-1]

        if attention_type == "none":
            self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.rgb_norm = nn.GroupNorm(16, self.feat_dim)
            self.aux_norm = nn.GroupNorm(16, self.feat_dim)
            self.input_proj = nn.Conv2d(in_channels - 3, 3, kernel_size=1)
            self.channel_attention = {
                "stream": MultiStreamAttention,
                "self": SelfAttentionModule,
                "cross": CrossAttentionModule
            }[attention_type](self.feat_dim)

    def forward(self, x):
        if self.attention_type == "none":
            x = self.input_proj(x)
            features = self.backbone(x)
            skips = features[:-1][::-1]  # Reverse for decoder use
            return features[-1], skips, None
        else:
            rgb_input, aux_input = x[:, :3], x[:, 3:]
            aux_input = self.input_proj(aux_input)

            if self.backbone_type == "swin":
                rgb_features = [f.permute(0, 3, 1, 2) for f in self.backbone(rgb_input)]
                aux_features = [f.permute(0, 3, 1, 2) for f in self.backbone(aux_input)]
            else:
                rgb_features = self.backbone(rgb_input)
                aux_features = self.backbone(aux_input)

            rgb_deep = rgb_features[-1] if self.backbone_type == "swin" else rgb_features[0]
            aux_deep = aux_features[-1] if self.backbone_type == "swin" else aux_features[0]

            rgb_deep = self.rgb_norm(rgb_deep)
            aux_deep = self.aux_norm(aux_deep)

            merged, _, attn_weights = self.channel_attention([
                rgb_deep, aux_deep
            ])

            if self.backbone_type == "swin":
                skips = rgb_features[:-1]
            else:
                skips = rgb_features[1:][::-1]
            return merged, skips, attn_weights


class KIIM(nn.Module):
    """
    Complete segmentation or classification model.

    Combines:
        - Backbone feature extractor (e.g., Swin Transformer)
        - Optional attention between RGB and auxiliary channels
        - Task-specific decoder (segmentation or classification)

    Args:
        model_name (str): Backbone model name.
        in_channels (int): Number of input channels.
        classes (int): Number of output classes.
        hidden_dim (int): Hidden dim (unused).
        encoder_name (str): Unused.
        encoder_weights (str): Unused.
        encoder_depth (int): Unused.
        decoder_attention_type (str): Unused.
        activation (str): Optional activation (unused).
        weights (str): Pretrained weights type for backbone.
        pretrained (bool): Load pretrained weights.
        attention_type (str): Type of attention ('none', 'stream', 'self', 'cross').
        task (str): 'segmentation' or 'classification'.

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W).

    Returns:
        logits (Tensor): Final prediction (segmentation map or class logits).
    """
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 16,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        encoder_depth: int = 5,
        decoder_attention_type: str = None,
        activation: str = None,
        attention_type: str = "self",
        task: str = "segmentation",
        **kwargs
    ):
        super().__init__()
        self.task = task

        # Initialize backbone and attention fusion wrapper
        self.backbone = create_backbone(model_name)
        self.attn_encoder = AttentionBackbone(self.backbone, attention_type, in_channels, model_name)

        # Choose appropriate decoder
        if self.task == 'segmentation':
            self.seg_decoder = SegmentationDecoder(self.attn_encoder.encoder_channels, num_classes)
        elif self.task == 'classification':
            self.cls_decoder = ClassificationDecoder(self.attn_encoder.feat_dim, num_classes)

    def forward(self, x):
        features, skips, attn_weights = self.attn_encoder(x)

        if self.task == "segmentation":
            logits = self.seg_decoder(features, skips)
        else:
            logits = self.cls_decoder(features)
        
        return logits
        # return {
        #     'logits': logits,
        #     # 'attention': attn_weights
        # }
