import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any, Union

from utils.losses import FocalLoss, DiceLoss, KGLoss
from utils.metrics import *
from model_v3.AttentionModule import LandUseMask
from model_v3.PretrainedModule_v1 import PretrainedModel
from model_v3.MultimodalImageryModule import MIM
from model_v3.ProjectionModule import ProjectionModule
from model_v3.VLModel_v2 import VLMResUNet

class IRRModel(nn.Module):
    """
    Knowledge-Informed Irrigation Mapping (KIIM) Model

    A modular segmentation model that supports multimodal inputs, attention mechanisms,
    knowledge-informed projections, and configurable loss functions. This model is 
    designed for agricultural asset mapping using remote sensing imagery.

    Args:
        BaseModel (Dict): Configuration dictionary for core model settings and flags.
            Expected keys:
                - use_pretrained_module (bool)
                - use_attention_module (bool)
                - use_multimodal_imagery_module (bool)
                - use_projection_module (bool)
                - loss_config (Dict[str, float])

        **kwargs: Additional module-specific configuration dictionaries.
            Optional keys:
                - PretrainedModule (Dict)
                - AttentionModule (Dict)
                - MultimodalImageryModule (Dict)
                - ProjectionModule (Dict)
    """

    def __init__(self, BaseModel, **kwargs):
        super().__init__()

        # Conditionally initialize submodules based on flags in BaseModel config
        if BaseModel['use_pretrained_module']:
            cfg = kwargs.get("PretrainedModule", {})
            self.pretrained_module = PretrainedModel(**cfg)

        if BaseModel['use_attention_module']:
            cfg = kwargs.get("AttentionModule", {})
            self.attention_module = LandUseMask(**cfg)

        if BaseModel['use_multimodal_imagery_module']:
            cfg = kwargs.get("MultimodalImageryModule", {})
            self.multimodal_imagery = MIM(**cfg)

        if BaseModel['use_projection_module']:
            cfg = kwargs.get("ProjectionModule", {})
            self.projection_module = ProjectionModule(**cfg)
        if BaseModel['use_vlm_module']:
            cfg = kwargs.get("VLMModule", {})
            self.pretrained_vlm = VLMResUNet(**cfg)

        # Loss configuration dictionary
        self.loss_config = BaseModel['loss_config']

        # Flags for conditional forward pass logic
        self.use_attention_module = BaseModel['use_attention_module']
        self.use_multimodal_imagery_module = BaseModel['use_multimodal_imagery_module']
        self.use_projection_module = BaseModel['use_projection_module']
        self.use_pretrained_module = BaseModel['use_pretrained_module']
        self.use_vlm_module = BaseModel['use_vlm_module']
        
        if self.use_pretrained_module and self.use_vlm_module:
            raise RuntimeError("Cannot use both pretrained and VLM module simultaneously. Please disable one.")

        if not(self.use_pretrained_module or self.use_vlm_module):
            raise RuntimeError("One of pretrained and VLM module should be enabled. Please enable one.")


        # Loss functions
        self.focal_loss = FocalLoss(gamma=2.0)  # You can parameterize gamma if needed
        self.ce_loss = FocalLoss(gamma=0)
        self.kg_loss = KGLoss()
        self.dice_loss = DiceLoss()

    def prepare_landmask(self, land_mask: torch.Tensor) -> torch.Tensor:
        """
        Binarize normalized land mask with tolerance for floating point comparisons.
        """
        return (
            torch.isclose(land_mask, torch.tensor(81.0 / 255.0), atol=1e-4) |
            torch.isclose(land_mask, torch.tensor(82.0 / 255.0), atol=1e-4)
        ).float()


    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the KIIM model.

        Args:
            batch (Dict): A batch of input tensors. Expected keys include:
                - 'land_mask'
                - 'crop_mask'
                - any inputs required by the MultimodalImageryModule

        Returns:
            Dict: Dictionary containing logits, predictions, and optionally attention and projection outputs.
        """
        output_dict = {}
        

        # Extract and process multimodal input features
        features = self.multimodal_imagery(batch)
        

        # Optional attention module
        if self.use_attention_module:
            land_mask = self.prepare_landmask(batch['land_mask'])
            # land_mask = batch['land_mask']
            att_out = self.attention_module(features, land_mask)
            features = att_out['features']

        # Pretrained classification head
        if self.use_pretrained_module:
            outputs = self.pretrained_module(features)
        elif self.use_vlm_module:
            outputs = self.pretrained_vlm(features, batch['text_prompt'])
        logits = outputs['logits']
        output_dict['logits'] = logits

        # Optional projection module for knowledge-guided refinement
        if self.use_projection_module and 'crop_mask' in batch:
            proj_out = self.projection_module(logits, batch['crop_mask'])
            output_dict['logits'] = proj_out
            del proj_out  # Free memory
            

        # Softmax predictions (non-differentiable)
        with torch.no_grad():
            output_dict['predictions'] = F.softmax(output_dict['logits'], dim=1)

        return output_dict

    def compute_loss(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        land_mask: Optional[torch.Tensor] = None,
        is_label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss using configurable weights for different loss functions.

        Args:
            logits (Tensor): Raw logits from the model.
            predictions (Tensor): Softmaxed predictions.
            targets (Tensor): Ground truth segmentation labels.
            land_mask (Optional[Tensor]): Mask indicating valid land regions.
            is_label (Optional[Tensor]): Mask indicating valid supervision regions.

        Returns:
            Dict[str, Tensor]: Dictionary of individual losses and the total loss.
        """
        losses = {}
        config = self.loss_config
        land_mask = self.prepare_landmask(land_mask)

        if config["ce_weight"] > 0:
            losses['ce_loss'] = self.focal_loss(logits, targets, is_label) * config["ce_weight"]

        if config["dice_weight"] > 0:
            losses['dice_loss'] = self.dice_loss(predictions, targets, land_mask, is_label) * config["dice_weight"]

        if config["focal_weight"] > 0:
            losses['focal_loss'] = self.focal_loss(logits, targets, is_label) * config["focal_weight"]

        if config["kg_weight"] > 0:
            # Make sure projection module is used before accessing its weights
            if hasattr(self, 'projection_module'):
                losses['kg_loss'] = self.kg_loss(self.projection_module.weights) * config["kg_weight"]

        # Final total loss
        losses['total_loss'] = sum(losses.values())
        return losses

    # Optional: Uncomment if training with PyTorch Lightning
    # def configure_optimizers(self) -> torch.optim.Optimizer:
    #     """
    #     Configure optimizers if using PyTorch Lightning.
    #     """
    #     return torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.learning_rate,
    #         weight_decay=self.weight_decay
    #     )
