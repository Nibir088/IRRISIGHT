import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Tuple, List, Optional, Dict
from models.ViTBackbone import *
from torchgeo.models import ResNet18_Weights, FarSeg
from models.FarSegModel import *
from models.SwinTransformer import *
def find_model(
    name: str,
    in_channels: int,
    classes: int,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    encoder_depth: int = 5,
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
    decoder_attention_type: Optional[str] = None,
    activation: Optional[str] = None
) -> nn.Module:
    """
    Factory function to create various segmentation model architectures.
    All models support the same set of parameters for consistency.
    
    Args:
        name (str): Model architecture name ('unet', 'resnet', 'manet', etc.)
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        classes (int): Number of output classes
        encoder_name (str, optional): Name of encoder backbone. Defaults to "resnet34"
        encoder_weights (str, optional): Pre-trained weights for encoder. Defaults to "imagenet"
        encoder_depth (int, optional): Depth of encoder. Defaults to 5
        decoder_channels (Tuple[int, ...], optional): Decoder channel dimensions. Defaults to (256, 128, 64, 32, 16)
        decoder_attention_type (Optional[str], optional): Type of attention in decoder. Defaults to None
        activation (Optional[str], optional): Final activation function. Defaults to None
    
    Returns:
        nn.Module: Initialized segmentation model
        
    Raises:
        ValueError: If invalid model name is provided
    """
    model_configs = {
        'unet': lambda: smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'manet': lambda: smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'linknet': lambda: smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            # decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'fpn': lambda: smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            # decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'psnet': lambda: smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            # decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'pan': lambda: smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            # encoder_depth=encoder_depth,
            # decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'deepv3': lambda: smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'deepv3+': lambda: smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            # encoder_depth=encoder_depth,
            # decoder_channels=decoder_channels,
            # decoder_attention_type=decoder_attention_type,
            activation=activation
        ),
        'vit': lambda: ViTSegmentation(
            backbone=timm.create_model('vit_small_patch16_224', pretrained=True),
            num_classes=classes,
            in_channels=in_channels,
            img_size=224
        ),
        # 'farseg': lambda: smp.Segformer(
        #     encoder_name=encoder_name,
        #     encoder_depth=encoder_depth,
        #     encoder_weights=encoder_weights,
        #     decoder_segmentation_channels=256,
        #     in_channels=in_channels,
        #     classes=classes,
        #     activation=activation
        # ),
        # 'farseg': lambda: FarSeg(
        #     backbone=encoder_name,
        #     classes=classes,
        #     backbone_pretrained=True
        # ),
        'farseg': lambda: FarSegModel(
            num_classes = classes,
            backbone_name = encoder_name,
            in_channels = in_channels
        ),
        'segformer': lambda: smp.Segformer(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        ),
        'swin': lambda: SwinUnet(
            num_classes=classes,
            in_channels=in_channels,
            backbone_name="swin_base_patch4_window7_224"
        )
        
    }
    
    if name.lower() not in model_configs:
        raise ValueError(f"Model {name} not supported. Available models: {list(model_configs.keys())}")
    
    return model_configs[name.lower()]()



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
        attention_type: Optional[str] = "self"
    ):
        super().__init__()
        
        # Create decoder channels tuple based on hidden_dim
        decoder_channels = tuple([hidden_dim * (2 ** i) for i in range(encoder_depth - 1, -1, -1)])
        
        self.model = find_model(
            name=model_name,
            in_channels=in_channels,
            classes=classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            activation=activation
        )
        self.is_vit = model_name.lower() == 'vit'
        self.is_swin = model_name.lower() == 'swin'
        
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
        if self.is_vit:
            # For ViT models
            features = self.model.backbone.forward_features(x)
        else:
            if hasattr(self.model, "encoder") and callable(getattr(self.model, "encoder", None)) and (not self.is_swin):
                features = self.model.encoder(x)
            else:
                features = None
            # For other models
            # features = self.model.encoder(x)
        logits = self.model(x)
        outputs = {
            'encoder_feature': features,
            'logits': logits
        }
            
        return outputs
