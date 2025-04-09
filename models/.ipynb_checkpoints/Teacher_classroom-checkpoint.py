# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any, Union
from utils.losses import *
from utils.metrics import *
from models.AttentionModule import *
from models.BackboneModel import *
# from models.multiStreamBackboneUpdated import *
# from models.multiStreamBackbonewSwin import *
from models.MultimodalImageryModule import *
from models.ProjectionModule import *
import copy
class KIIM(nn.Module):
    """
    Knowledge Informed Irrigation Mapping (KIIM) model implemented in PyTorch Lightning.
    
    Args:
        backbone_name (str): Name of the backbone model ('unet', 'resnet', etc.)
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels
        learning_rate (float): Learning rate for optimization
        use_attention (bool): Whether to use the attention module
        use_mim (bool): Whether to use the multimodal imagery module
        use_projection (bool): Whether to use the projection module
        use_ensemble (bool): Whether to use the ensemble module
        hidden_dim (int): Hidden dimension for intermediate features
        weight_decay (float): Weight decay for optimization
        **kwargs: Additional arguments for the backbone model
        loss_config (Dict): Configuration for loss functions with weights
        
    """
    def __init__(
        self,
        backbone_name: str = "resnet",
        encoder_name: str = 'resnet152',
        num_classes: int = 4,
        learning_rate: float = 1e-4,
        use_attention: bool = True,
        use_projection: bool = True,
        use_rgb: bool = True,
        use_vegetation: bool = True,
        use_ndvi: bool = True,
        use_ndwi: bool = True,
        use_ndti: bool = True,
        pretrained_hidden_dim: int = 16,
        attention_hidden_dim: int = 16,
        gamma: float = 5.0,
        task: str = 'segmentation',
        weight_decay: float = 1e-4,
        loss_config: Dict[str, float] = {
            "ce_weight": 1.0,
            "dice_weight": 0.0,
            "focal_weight": 0.0,
            "kg_weight": 0.0,
            "stream_weight": 0.0
        },
        **kwargs
    ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize modules based on configuration
        if use_attention:
            self.attention = LandUseMask(in_channels=1, hidden_dim=attention_hidden_dim)
        
        self.mim = MIM(
                use_rgb=use_rgb,
                use_vegetation=use_vegetation,
                use_ndvi=use_ndvi,
                use_ndwi=use_ndwi,
                use_ndti=use_ndti
        )
        in_channels = self.mim.total_channels
            
        # Initialize backbone model
        self.backbone = PretrainedModel(
            model_name=backbone_name,
            in_channels=in_channels,
            classes=num_classes,
            hidden_dim=pretrained_hidden_dim,
            encoder_name = encoder_name,
            **kwargs
        )

        self.loss_config = loss_config
        
        if use_projection:
            self.projection = ProjectionModule(num_classes=num_classes, in_channels=21)
            
        self.use_attention = use_attention
        self.use_projection = use_projection
        
        self.focal_loss = FocalLoss(gamma=gamma)
        self.ce_loss = FocalLoss(gamma=0)
        self.kg_loss = KGLoss()
        self.dice_loss = DiceLoss()
        
        
        self.classes = num_classes
        
        
#     def on_train_epoch_start(self):
#         """Reset metrics at the start of training epoch."""
#         self.train_metrics.reset()

#     def on_validation_epoch_start(self):
#         """Reset metrics at the start of validation epoch."""
#         self.val_metrics.reset()
#     # def on_test_epoch_start(self):
#     #     """Reset metrics at the start of test epoch."""
#     #     self.test_metrics.reset()

    def prepare_landmask(self, land_mask: torch.Tensor) -> torch.Tensor:
        return ((land_mask == 1) | (land_mask == 2)).float()
    def forward(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the KIIM model.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch containing:
                - 'image': RGB image [B, 3, H, W]
                - 'ndvi': NDVI index [B, H, W]
                - 'ndwi': NDWI index [B, H, W]
                - 'ndti': NDTI index [B, H, W]
                - 'land_mask': Land use mask [B, 1, H, W]
                - 'crop_mask': Crop mask [B, num_classes, H, W]
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model outputs
        """
        
        output_dict = {}
        x = batch
        # Apply Multimodal Imagery Module
        features = self.mim(x)
        
        
        land_mask = self.prepare_landmask(x['land_mask'])
        # Apply Attention Module
        if self.use_attention:
            output = self.attention(features, land_mask)
            features = output['features']
            output_dict['attention'] = output['attention']
            output_dict['AM_features'] = features
            
        # Get backbone predictions
        outputs = self.backbone(features)
        
        logits = outputs['logits']
        output_dict['PM_logits'] = logits
        output_dict['encoder_feature'] = None#outputs['encoder_feature'][-1]
        output_dict['logits'] = logits
        
        if 'stream_pred' in outputs.keys():
            output_dict['stream_pred'] = outputs['stream_pred']
        
        
        # Apply Projection Module
        # if self.use_projection:
        #     logits = self.projection(logits, x['crop_mask'])
        #     output_dict['CPM_logits'] = logits
        #     output_dict['logits'] = logits
        # Apply final softmax
        predictions = F.softmax(logits, dim=1)
        output_dict['predictions'] = predictions
        
        return output_dict
    

    def compute_loss(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        land_mask: Optional[torch.Tensor] = None,
        stream_pred: Optional[torch.Tensor] = None,
        is_label = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss with all components.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            knowledge_mask (Optional[torch.Tensor]): Knowledge guidance mask
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing individual and total losses
        """
        
        losses = {}
        if self.loss_config["ce_weight"]>0:
            losses['ce_loss'] = self.focal_loss(logits, targets, is_label) * self.loss_config["ce_weight"]
        
        if self.loss_config["dice_weight"]>0:
            losses['dice_loss'] = self.dice_loss(predictions, targets, land_mask, is_label) * self.loss_config["dice_weight"]
        if self.loss_config["focal_weight"]>0:
            losses['focal_loss'] = self.focal_loss(logits, targets, is_label) * self.loss_config["focal_weight"]
        if self.loss_config["kg_weight"]>0:
            losses['kg_loss'] = self.kg_loss(self.projection.weights) * self.loss_config["kg_weight"]
        if stream_pred!=None and self.loss_config["stream_weight"]>0:
            losses['stream_loss'] = (self.ce_loss(stream_pred[1], targets)+self.ce_loss(stream_pred[0], targets)) * self.loss_config["stream_weight"]
            
        # Compute total loss
        losses['total_loss'] = sum(losses.values())
        return losses
    # def on_train_epoch_end(self):
    #     """
    #     Compute and log training metrics at the end of epoch.
    #     """
    #     metrics = self.train_metrics.compute()
    #     for metric_name, metric_values in metrics.items():
    #         for avg_type, value in metric_values.items():
    #             if avg_type != 'per_class':
    #                 self.log(f'train_{metric_name}_{avg_type}', value, sync_dist=True)
    #             else:
    #                 # Log per-class metrics
    #                 for class_idx, class_value in enumerate(value):
    #                     self.log(f'train_{metric_name}_class_{class_idx}', class_value, sync_dist=True)

                    
#     def training_step(
#         self,
#         batch: Dict[str, torch.Tensor],
#         batch_idx: int
#     ) -> torch.Tensor:
#         """
#         Training step with multiple losses and metrics tracking.
#         """
        
#         outputs = self(batch)
#         losses = self.compute_loss(
#             outputs['logits'],
#             outputs['predictions'],
#             batch['true_mask'],
#             batch.get('land_mask', None),
#             outputs.get('stream_pred',None)
#         )
#         # Update metrics
#         self.train_metrics.update(
#             batch['true_mask'].detach().cpu().numpy(),
#             outputs['predictions'].detach().cpu().numpy()
#         )
        
#         # Log all losses
#         for loss_name, loss_value in losses.items():
#             self.log(f'train_{loss_name}', loss_value,sync_dist=True)
            
#         return losses['total_loss']
        
    
#     def validation_step(
#         self,
#         batch: Dict[str, torch.Tensor],
#         batch_idx: int
#     ) -> None:
#         """
#         Validation step with multiple losses.
        
#         Args:
#             batch (Dict[str, torch.Tensor]): Input batch
#             batch_idx (int): Index of current batch
#         """
#         outputs = self(batch)
#         losses = self.compute_loss(
#             outputs['logits'],
#             outputs['predictions'],
#             batch['true_mask'],
#             batch.get('land_mask', None),
#             outputs.get('stream_pred',None)
#         )
        
#         # Update metrics
#         self.val_metrics.update(
#             batch['true_mask'].detach().cpu().numpy(),
#             outputs['predictions'].detach().cpu().numpy()
#         )
        
#         # Log all losses
#         for loss_name, loss_value in losses.items():
#             self.log(f'val_{loss_name}', loss_value, sync_dist=True)

#     def on_validation_epoch_end(self):
#         """
#         Compute and log validation metrics at the end of epoch.
#         """
#         metrics = self.val_metrics.compute()
#         for metric_name, metric_values in metrics.items():
#             for avg_type, value in metric_values.items():
#                 if avg_type != 'per_class':
#                     self.log(f'val_{metric_name}_{avg_type}', value, sync_dist=True)
                    
#                 else:
#                     # Log per-class metrics separately
#                     for class_idx, class_value in enumerate(value):
#                         self.log(f'val_{metric_name}_class_{class_idx}', class_value, sync_dist=True)
#                         print(f'val_{metric_name}_class_{class_idx}', class_value)
#         # print(f'Weights: ', self.projection.weights)
    
#     def test_step(
#         self,
#         batch: Dict[str, torch.Tensor],
#         batch_idx: int,
#         dataloader_idx: str = None
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Testing step with multiple metrics and loss calculations.
#         Args:
#             batch (Dict[str, torch.Tensor]): Input batch
#             batch_idx (int): Index of current batch
#             dataloader_idx (str): Identifier for the current test dataloader
#         """
#         # Ensure dataloader_idx is set
#         dataloader_idx = dataloader_idx if dataloader_idx is not None else "default"

#         outputs = self(batch)
#         # Compute losses
#         losses = self.compute_loss(
#             outputs['logits'],
#             outputs['predictions'],
#             batch['true_mask'],
#             batch.get('land_mask', None),
#             outputs.get('stream_pred', None)
#         )

#         # Initialize test metrics for each dataloader on first use
#         if dataloader_idx not in self.test_metrics_dict:
#             self.test_metrics_dict[dataloader_idx] = SegmentationMetrics(self.classes)

#         # Update metrics for current dataloader
#         self.test_metrics_dict[dataloader_idx].update(
#             batch['true_mask'].detach().cpu().numpy(),
#             outputs['predictions'].detach().cpu().numpy()
#         )

#         # Log losses with dataloader-specific prefix
#         for loss_name, loss_value in losses.items():
#             self.log(f'test_{dataloader_idx}_{loss_name}', loss_value, sync_dist=True)


#     def on_test_epoch_end(self, dataloader_idx: str = None):
#         """
#         Aggregates and logs metrics at the end of testing for each dataloader.
#         """
#         if not hasattr(self, 'test_metrics_dict') or not self.test_metrics_dict:
#             raise ValueError("No test metrics available. Ensure test_step is implemented correctly.")
        
        
        
        
#         # Process metrics for each dataloader we've seen
#         for dataloader_idx, metrics_calculator in self.test_metrics_dict.items():
#             metrics = metrics_calculator.compute()

#             # Log metrics with dataloader-specific prefix
#             for metric_name, metric_values in metrics.items():
#                 for avg_type, value in metric_values.items():
#                     if avg_type != 'per_class':
#                         self.log(f'test_{dataloader_idx}_{metric_name}_{avg_type}', 
#                                  value, sync_dist=True)
#                     else:
#                         # Log per-class metrics
#                         for class_idx, class_value in enumerate(value):
#                             self.log(f'test_{dataloader_idx}_{metric_name}_class_{class_idx}', 
#                                      class_value, sync_dist=True)
#             # Log summary to console for better debugging (optional)
#             print(f"\nTest metrics for {dataloader_idx}: {metrics}")

#             # Reset metrics for next test
#             # metrics_calculator.reset()


                    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model specific arguments to ArgumentParser.
        
        Args:
            parent_parser: Parent ArgumentParser
            
        Returns:
            Updated ArgumentParser
        """
        parser = parent_parser.add_argument_group("KIIM")
        parser.add_argument("--backbone_name", type=str, default="resnet")
        parser.add_argument("--num_classes", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--use_attention", type=bool, default=True)
        parser.add_argument("--use_projection", type=bool, default=True)
        parser.add_argument("--pretrained_hidden_dim", type=int, default=16)
        parser.add_argument("--attention_hidden_dim", type=int, default=16)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--use_rgb", type=bool, default=True)
        parser.add_argument("--use_ndvi", type=bool, default=True)
        parser.add_argument("--use_ndwi", type=bool, default=True)
        parser.add_argument("--use_ndti", type=bool, default=True)
        parser.add_argument("--use_vegetation", type=bool, default=True)
        
        parser.add_argument("--ce_weight", type=float, default=0.0)
        parser.add_argument("--dice_weight", type=float, default=0.35)
        parser.add_argument("--focal_weight", type=float, default=0.45)
        parser.add_argument("--kg_weight", type=float, default=0.2)
        parser.add_argument("--gamma", type=float, default=2.0)
        parser.add_argument("--task", type=str, default='segmentation')
        return parent_parser
#     def get_metrics(self) -> Dict[str, Any]:
#         """
#         Get consolidated metrics from training, validation, and test sets.

#         Returns:
#             Dict[str, Any]: Dictionary containing metrics organized by:
#                 - train_metrics: Training metrics
#                 - val_metrics: Validation metrics
#                 - test_results: Test metrics (state-wise if applicable)
#         """
#         # Get computed metrics from both training and validation
#         train_metrics_raw = self.train_metrics.compute()
#         val_metrics_raw = self.val_metrics.compute()

#         # Initialize the metrics dictionary
#         final_metrics = {
#             "train_metrics": {
#                 "overall": {},
#                 "per_class": {},
#                 "losses": {}
#             },
#             "val_metrics": {
#                 "overall": {},
#                 "per_class": {},
#                 "losses": {}
#             },
#             "test_results": {}  # Add placeholder for test results
#         }

#         # Helper function to organize metrics
#         def organize_metrics(raw_metrics, prefix, target_dict):
#             for metric_name, metric_values in raw_metrics.items():
#                 for avg_type, value in metric_values.items():
#                     if avg_type == 'per_class':
#                         # Handle per-class metrics
#                         target_dict["per_class"][metric_name] = {
#                             f"class_{i}": float(v)
#                             for i, v in enumerate(value)
#                         }
#                     else:
#                         # Handle overall metrics (micro, macro, weighted)
#                         target_dict["overall"][f"{metric_name}_{avg_type}"] = float(value)

#         # Organize training metrics
#         organize_metrics(
#             train_metrics_raw,
#             "train",
#             final_metrics["train_metrics"]
#         )

#         # Organize validation metrics
#         organize_metrics(
#             val_metrics_raw,
#             "val",
#             final_metrics["val_metrics"]
#         )

#         # Add loss components for both training and validation
# #         loss_components = [
# #             "ce_loss", "dice_loss", "focal_loss",
# #             "kg_loss", "stream_loss", "total_loss"
# #         ]

# #         for component in loss_components:
# #             # Get the latest logged values for each loss component
# #             train_loss = self.trainer.callback_metrics.get(
# #                 f"train_{component}", np.NaN
# #             )
# #             val_loss = self.trainer.callback_metrics.get(
# #                 f"val_{component}", np.NaN
# #             )

# #             final_metrics["train_metrics"]["losses"][component] = float(train_loss)
# #             final_metrics["val_metrics"]["losses"][component] = float(val_loss)
        
#         # Organize test results (state-wise if applicable)
#         if hasattr(self, "test_metrics_dict"):
#             for state, metrics_calculator in self.test_metrics_dict.items():
#                 state_metrics_raw = metrics_calculator.compute()
#                 print(state_metrics_raw)
#                 state_metrics = {
#                     "overall": {},
#                     "per_class": {}
#                 }
#                 organize_metrics(
#                     state_metrics_raw,
#                     f"test_{state}",
#                     state_metrics
#                 )
#                 final_metrics["test_results"][state] = state_metrics

#         return final_metrics


