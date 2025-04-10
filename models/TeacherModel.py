import pytorch_lightning as pl
import torch
from typing import Any, Dict
from models.KIIM import KIIM  # Assuming your base model is in models/KIIM.py

class KIIMLightning(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str = "resnet",
        encoder_name: str = 'resnet152',
        num_classes: int = 4,
        learning_rate: float = 1e-4,
        use_attention: bool = True,
        use_projection: bool = True,
        use_rgb: bool = True,
        use_ndvi: bool = True,
        use_ndwi: bool = True,
        use_ndti: bool = True,
        pretrained_hidden_dim: int = 16,
        attention_hidden_dim: int = 16,
        gamma: float = 5.0,
        weight_decay: float = 1e-4,
        loss_config: Dict[str, float] = {
            "ce_weight": 0.0,
            "dice_weight": 0.25,
            "focal_weight": 0.35,
            "kg_weight": 0.2,
            "stream_weight": 0.2
        },
        **kwargs
    ):
        super().__init__()
        self.model = KIIM(
            backbone_name=backbone_name,
            encoder_name=encoder_name,
            num_classes=num_classes,
            learning_rate=learning_rate,
            use_attention=use_attention,
            use_projection=use_projection,
            use_rgb=use_rgb,
            use_ndvi=use_ndvi,
            use_ndwi=use_ndwi,
            use_ndti=use_ndti,
            pretrained_hidden_dim=pretrained_hidden_dim,
            attention_hidden_dim=attention_hidden_dim,
            gamma=gamma,
            weight_decay=weight_decay,
            loss_config=loss_config,
            **kwargs
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        losses = self.model.compute_loss(
            outputs['logits'],
            outputs['predictions'],
            batch['true_mask'],
            batch.get('land_mask', None),
            outputs.get('stream_pred', None),
            batch.get('is_label', None)
        )
        for loss_name, loss_value in losses.items():
            self.log(f'train_{loss_name}', loss_value, sync_dist=True)
        return losses['total_loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self(batch)
        losses = self.model.compute_loss(
            outputs['logits'],
            outputs['predictions'],
            batch['true_mask'],
            batch.get('land_mask', None),
            outputs.get('stream_pred', None),
            batch.get('is_label', None)
        )
        for loss_name, loss_value in losses.items():
            self.log(f'val_{loss_name}', loss_value, sync_dist=True)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    @staticmethod
    def add_model_specific_args(parent_parser):
        return KIIM.add_model_specific_args(parent_parser)
