import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import segmentation_models_pytorch as smp
import timm

# Custom imports
from model_v3.ViTBackbone import ViTSegmentation
from model_v3.SwinTransformer import SwinUnet
from model_v3.FarSegModel import FarSegModel
from model_v3.KIIM import KIIM
from model_v3.SAM import SAMSegmentation


def find_model(
    name: str,
    in_channels: int,
    classes: int,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    decoder_channels: Optional[Tuple[int, ...]] = None,
    activation: Optional[str] = None,
    hidden_dim: int = 16,
    attention_type: str = "self",
    task: str = "segmentation",
    freeze_model: bool = False
) -> nn.Module:
    """
    Create and return a segmentation model instance.

    Args:
        name (str): Model name.
        in_channels (int): Input channels.
        classes (int): Output classes.
        encoder_name (str): Encoder backbone name.
        encoder_weights (str): Pretrained encoder weights.
        decoder_channels (Tuple[int]): Decoder channels.
        activation (str): Final activation.
        hidden_dim (int): For KIIM model.
        attention_type (str): For KIIM attention fusion.
        task (str): Task type ('segmentation' or 'classification').

    Returns:
        nn.Module: Instantiated model.
    """
    if name.lower() == 'kiim':
        return KIIM(
            model_name=encoder_name,
            in_channels=in_channels,
            num_classes=classes,
            hidden_dim=hidden_dim,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            activation=activation,
            attention_type=attention_type,
            task=task
        )
    
    if name.lower() == 'sam':
        return SAMSegmentation(
            num_classes=classes,
            freeze_model = freeze_model
        )

    # Set default decoder_channels if not provided
    if decoder_channels is None:
        decoder_channels = tuple(hidden_dim * (2 ** i) for i in reversed(range(5)))

    model_configs = {
        'unet': smp.Unet,
        'fpn': smp.FPN,
        'deepv3+': smp.DeepLabV3Plus,
        'segformer': smp.Segformer
    }

    name = name.lower()

    if name in model_configs:
        return model_configs[name](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            # decoder_channels=decoder_channels,
            activation=activation
        )
    elif name == 'vit':
        return ViTSegmentation(
            backbone=timm.create_model('vit_small_patch16_224', pretrained=True),
            num_classes=classes,
            in_channels=in_channels,
            img_size=224
        )
    elif name == 'swin':
        return SwinUnet(
            num_classes=classes,
            in_channels=in_channels,
            backbone_name="swin_base_patch4_window7_224"
        )
    elif name == 'farseg':
        return FarSegModel(
            num_classes=classes,
            backbone_name=encoder_name,
            in_channels=in_channels
        )
    else:
        raise ValueError(f"Model '{name}' not supported. Available: unet, fpn, deepv3+, vit, swin, farseg, kiim")


class PretrainedModel(nn.Module):
    """
    Unified segmentation model wrapper for different architectures.

    Combines:
        - Backbone feature extractor
        - Task-specific decoder (segmentation/classification)

    Args:
        model_name (str): One of [unet, fpn, deepv3+, vit, swin, farseg, kiim]
        in_channels (int): Number of input channels (e.g., 3 + N)
        num_classes (int): Output classes
        hidden_dim (int): Base hidden size (default 16)
        encoder_name (str): Backbone name (e.g., 'resnet34', 'resnet50')
        encoder_weights (str): Pretrained weights (e.g., 'imagenet')
        activation (str): Output activation function
        attention_type (str): Attention mode for KIIM only ['self', 'cross', 'none']
        task (str): Task type ('segmentation' or 'classification')
    """
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 16,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        activation: Optional[str] = None,
        attention_type: str = "self",
        task: str = "segmentation",
        freeze_model: bool = True
    ):
        super().__init__()
        self.model = find_model(
            name=model_name,
            in_channels=in_channels,
            classes=num_classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            hidden_dim=hidden_dim,
            activation=activation,
            attention_type=attention_type,
            task=task,
            freeze_model = freeze_model
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward input through the model.

        Args:
            x (Tensor): Input tensor (B, C, H, W)

        Returns:
            Dict[str, Tensor]: {
                'logits': prediction map
            }
        """
        if isinstance(self.model(x), dict):  # KIIM returns dict
            return self.model(x)
        else:
            return {'logits': self.model(x)}
