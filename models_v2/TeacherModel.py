import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from models_v2.BaseModel import *
from utils.metrics import *

class TeacherModel(pl.LightningModule):
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg['general']
        self.teacher = KIIM(**cfg['model'])
        num_classes = cfg['general']['num_classes']
        
        self.train_metrics = SegmentationMetrics(num_classes)
        self.val_metrics = SegmentationMetrics(num_classes)
        # self.test_metrics = SegmentationMetrics(num_classes)
        self.test_metrics_dict = {}
        
        self.classes = num_classes
        # for param in self.teacher.parameters():
        #     param.requires_grad = False
        
    
    def on_train_epoch_start(self):
        """Reset metrics at the start of training epoch."""
        self.train_metrics.reset()

    def on_validation_epoch_start(self):
        """Reset metrics at the start of validation epoch."""
        self.val_metrics.reset()
    # def on_test_epoch_start(self):
    #     """Reset metrics at the start of test epoch."""
    #     self.test_metrics.reset()

    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.teacher(batch)
        
        return output
    
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        
        # Separate labeled and unlabeled data using masks
        is_labeled = batch['is_labeled'].bool()

        # Compute labeled loss only for labeled data
        labeled_loss = 0.0
        if torch.any(is_labeled):
            labeled_logits = outputs['logits']
            # print(labeled_logits.shape)
            labeled_predictions = outputs['predictions']
            labeled_targets = batch['true_mask']
            
            # print(batch['land_mask'].shape)
            labeled_loss = self.teacher.compute_loss(labeled_logits, labeled_predictions, labeled_targets, land_mask = batch['land_mask'],is_label = is_labeled)['total_loss']

        
        
        self.train_metrics.update(
            batch['true_mask'].detach().cpu().numpy(),
            outputs['predictions'].detach().cpu().numpy()
        )
        
            
        self.log("train_total_loss", labeled_loss, sync_dist=True, on_step=True, on_epoch=True)

        return labeled_loss
    
    def on_train_epoch_end(self):
        """
        Compute and log training metrics at the end of epoch.
        """
        metrics = self.train_metrics.compute()
        for metric_name, metric_values in metrics.items():
            for avg_type, value in metric_values.items():
                if avg_type != 'per_class':
                    self.log(f'train_{metric_name}_{avg_type}', value, sync_dist=True)
                else:
                    # Log per-class metrics
                    for class_idx, class_value in enumerate(value):
                        self.log(f'train_{metric_name}_class_{class_idx}', class_value, sync_dist=True)
                        

    def configure_optimizers(self):
        return torch.optim.AdamW(self.teacher.parameters(), lr=self.cfg['learning_rate'])
    
    
    
            
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step with multiple losses.
        
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            batch_idx (int): Index of current batch
        """
        outputs = self.teacher(batch)
        labeled_loss = self.teacher.compute_loss(
            outputs['logits'],
            outputs['predictions'],
            batch['true_mask'],
            land_mask = batch['land_mask'],is_label = batch['is_labeled']
        )['total_loss']
        # Update metrics
        self.val_metrics.update(
            batch['true_mask'].detach().cpu().numpy(),
            outputs['predictions'].detach().cpu().numpy()
        )
        self.log("val_loss", labeled_loss, sync_dist=True, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        """
        Compute and log validation metrics at the end of epoch.
        """
        metrics = self.val_metrics.compute()
        for metric_name, metric_values in metrics.items():
            for avg_type, value in metric_values.items():
                if avg_type != 'per_class':
                    self.log(f'val_{metric_name}_{avg_type}', value, sync_dist=True)
                    
                else:
                    # Log per-class metrics separately
                    for class_idx, class_value in enumerate(value):
                        self.log(f'val_{metric_name}_class_{class_idx}', class_value, sync_dist=True)
                        print(f'val_{metric_name}_class_{class_idx}', class_value)
        # print(f'Weights: ', self.projection.weights)
        

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: str = None):
        """
        Testing step with multiple metrics and loss calculations.
        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            batch_idx (int): Index of current batch
            dataloader_idx (str): Identifier for the current test dataloader
        """
        dataloader_idx = dataloader_idx if dataloader_idx is not None else "default"
        
        outputs = self.teacher(batch)
        labeled_loss = self.teacher.compute_loss(
            outputs['logits'],
            outputs['predictions'],
            batch['true_mask'],
            land_mask = batch['land_mask'],is_label = batch['is_labeled']
        )['total_loss']
        self.log("test_loss", labeled_loss, sync_dist=True,  on_step=False, on_epoch=True)
        # Initialize test metrics for each dataloader on first use
        if dataloader_idx not in self.test_metrics_dict:
            self.test_metrics_dict[dataloader_idx] = SegmentationMetrics(self.classes)

        # Update metrics for current dataloader
        self.test_metrics_dict[dataloader_idx].update(
            batch['true_mask'].detach().cpu().numpy(),
            outputs['predictions'].detach().cpu().numpy()
        )

        # Log losses with dataloader-specific prefix
        # for loss_name, loss_value in losses.items():
        #     self.log(f'test_{dataloader_idx}_{loss_name}', loss_value, sync_dist=True)

        
      

        

    def on_test_epoch_end(self, dataloader_idx: str = None):
        """
        Aggregates and logs metrics at the end of testing for each dataloader.
        """
        
        dataloader_idx = dataloader_idx if dataloader_idx is not None else "default"

        if not hasattr(self, 'test_metrics_dict') or not self.test_metrics_dict:
            raise ValueError("No test metrics available. Ensure test_step is implemented correctly.")
        
        
        
        
        # Process metrics for each dataloader we've seen
        for dataloader_idx, metrics_calculator in self.test_metrics_dict.items():
            metrics = metrics_calculator.compute()

            # Log metrics with dataloader-specific prefix
            for metric_name, metric_values in metrics.items():
                for avg_type, value in metric_values.items():
                    if avg_type != 'per_class':
                        self.log(f'test_{dataloader_idx}_{metric_name}_{avg_type}', 
                                 value, sync_dist=True)
                    else:
                        # Log per-class metrics
                        for class_idx, class_value in enumerate(value):
                            self.log(f'test_{dataloader_idx}_{metric_name}_class_{class_idx}', 
                                     class_value, sync_dist=True)
            # Log summary to console for better debugging (optional)
            print(f"\nTest metrics for {dataloader_idx}: {metrics}")

            # Reset metrics for next test
            # metrics_calculator.reset()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get consolidated metrics from training, validation, and test sets.

        Returns:
            Dict[str, Any]: Dictionary containing metrics organized by:
                - train_metrics: Training metrics
                - val_metrics: Validation metrics
                - test_results: Test metrics (state-wise if applicable)
        """
        # Get computed metrics from both training and validation
        train_metrics_raw = self.train_metrics.compute()
        val_metrics_raw = self.val_metrics.compute()

        # Initialize the metrics dictionary
        final_metrics = {
            "train_metrics": {
                "overall": {},
                "per_class": {},
                "losses": {}
            },
            "val_metrics": {
                "overall": {},
                "per_class": {},
                "losses": {}
            },
            "test_results": {}  # Add placeholder for test results
        }

        # Helper function to organize metrics
        def organize_metrics(raw_metrics, prefix, target_dict):
            for metric_name, metric_values in raw_metrics.items():
                for avg_type, value in metric_values.items():
                    if avg_type == 'per_class':
                        # Handle per-class metrics
                        target_dict["per_class"][metric_name] = {
                            f"class_{i}": float(v)
                            for i, v in enumerate(value)
                        }
                    else:
                        # Handle overall metrics (micro, macro, weighted)
                        target_dict["overall"][f"{metric_name}_{avg_type}"] = float(value)

        # Organize training metrics
        organize_metrics(
            train_metrics_raw,
            "train",
            final_metrics["train_metrics"]
        )

        # Organize validation metrics
        organize_metrics(
            val_metrics_raw,
            "val",
            final_metrics["val_metrics"]
        )

        # Add loss components for both training and validation
#         loss_components = [
#             "ce_loss", "dice_loss", "focal_loss",
#             "kg_loss", "stream_loss", "total_loss"
#         ]

#         for component in loss_components:
#             # Get the latest logged values for each loss component
#             train_loss = self.trainer.callback_metrics.get(
#                 f"train_{component}", np.NaN
#             )
#             val_loss = self.trainer.callback_metrics.get(
#                 f"val_{component}", np.NaN
#             )

#             final_metrics["train_metrics"]["losses"][component] = float(train_loss)
#             final_metrics["val_metrics"]["losses"][component] = float(val_loss)
        
        # Organize test results (state-wise if applicable)
        if hasattr(self, "test_metrics_dict"):
            for state, metrics_calculator in self.test_metrics_dict.items():
                state_metrics_raw = metrics_calculator.compute()
                print(state_metrics_raw)
                state_metrics = {
                    "overall": {},
                    "per_class": {}
                }
                organize_metrics(
                    state_metrics_raw,
                    f"test_{state}",
                    state_metrics
                )
                final_metrics["test_results"][state] = state_metrics

        return final_metrics


