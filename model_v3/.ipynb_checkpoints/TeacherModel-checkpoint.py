import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from model_v3.BaseModel import *
from utils.metrics import *

class TeacherModel(pl.LightningModule):
    def __init__(self, **cfg):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg['general']
        self.teacher = IRRModel(**cfg['teachermodel'])
        num_classes = cfg['general']['num_classes']
        
        self.target = cfg['teachermodel']['target']
        
        self.train_metrics = SegmentationMetrics(num_classes)
        self.val_metrics = SegmentationMetrics(num_classes)
        self.test_metrics = SegmentationMetrics(num_classes)

        
        self.classes = num_classes
        
    
    def on_train_epoch_start(self):
        """Reset metrics at the start of training epoch."""
        self.train_metrics.reset()

    def on_validation_epoch_start(self):
        """Reset metrics at the start of validation epoch."""
        self.val_metrics.reset()
    def on_test_epoch_start(self):
        """Reset metrics at the start of test epoch."""
        self.test_metrics.reset()

    
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
            labeled_targets = batch[self.target]#.argmax(dim=1)
            
            # print(batch['land_mask'].shape)
            labeled_loss = self.teacher.compute_loss(labeled_logits, labeled_predictions, labeled_targets, land_mask = batch['land_mask'],is_label = is_labeled)['total_loss']

        
        
        self.train_metrics.update(
            batch[self.target].detach().cpu().numpy(), #.argmax(dim=1)
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
            batch[self.target], #.argmax(dim=1)
            land_mask = batch['land_mask'],is_label = batch['is_labeled']
        )['total_loss']
        # Update metrics
        self.val_metrics.update(
            batch[self.target].detach().cpu().numpy(), #.argmax(dim=1)
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
        

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Test step for evaluating performance on held-out data.

        Args:
            batch (Dict[str, torch.Tensor]): Input batch
            batch_idx (int): Index of the batch
        """
        outputs = self.teacher(batch)
        
#         self.cfg.unsupervised:
#             multiclass_preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()  # [B, H, W]
#             binary_preds = (multiclass_preds > 0).astype("uint8")

#             polygons = [wkt.loads(p) for p in batch["polygon"]]  # convert from string to shapely

#             for i, polygon in enumerate(polygons):
#                 pred_mask = binary_preds[i]
#                 height, width = pred_mask.shape
#                 minx, miny, maxx, maxy = polygon.bounds
#                 transform = from_bounds(minx, miny, maxx, maxy, width, height)

#                 out_path = f"predictions/test_{batch_idx}_{i}.tif"
#                 os.makedirs(os.path.dirname(out_path), exist_ok=True)

#                 with rasterio.open(
#                     out_path, "w",
#                     driver="GTiff",
#                     height=height,
#                     width=width,
#                     count=1,
#                     dtype="uint8",
#                     crs=CRS.from_epsg(5070),
#                     transform=transform
#                 ) as dst:
#                     dst.write(pred_mask, 1)
#                 return 

        labeled_loss = self.teacher.compute_loss(
            outputs['logits'],
            outputs['predictions'],
            batch[self.target],
            land_mask=batch['land_mask'],
            is_label=batch['is_labeled']
        )['total_loss']

        self.test_metrics.update(
            batch[self.target].detach().cpu().numpy(),
            outputs['predictions'].detach().cpu().numpy()
        )

        self.log("test_loss", labeled_loss, sync_dist=True, on_step=False, on_epoch=True)


    def on_test_epoch_end(self):
        """
        Compute and log test metrics at the end of test epoch.
        """
        metrics = self.test_metrics.compute()

        for metric_name, metric_values in metrics.items():
            for avg_type, value in metric_values.items():
                if avg_type != 'per_class':
                    self.log(f'test_{metric_name}_{avg_type}', value, sync_dist=True)
                else:
                    for class_idx, class_value in enumerate(value):
                        self.log(f'test_{metric_name}_class_{class_idx}', class_value, sync_dist=True)
                        print(f'test_{metric_name}_class_{class_idx}', class_value)


    def get_metrics(self) -> Dict[str, Any]:
        """
        Get consolidated metrics from training, validation, and test sets.

        Returns:
            Dict[str, Any]: Dictionary containing metrics organized by:
                - train_metrics
                - val_metrics
                - test_metrics
        """
        # Fetch computed metrics from each phase
        train_metrics_raw = self.train_metrics.compute()
        val_metrics_raw = self.val_metrics.compute()
        test_metrics_raw = self.test_metrics.compute()

        final_metrics = {
            "train_metrics": {"overall": {}, "per_class": {}, "losses": {}},
            "val_metrics": {"overall": {}, "per_class": {}, "losses": {}},
            "test_metrics": {"overall": {}, "per_class": {}, "losses": {}},
        }

        # Helper function to organize into overall/per_class format
        def organize_metrics(raw_metrics, target_dict):
            for metric_name, metric_values in raw_metrics.items():
                for avg_type, value in metric_values.items():
                    if avg_type == 'per_class':
                        target_dict["per_class"][metric_name] = {
                            f"class_{i}": float(v) for i, v in enumerate(value)
                        }
                    else:
                        target_dict["overall"][f"{metric_name}_{avg_type}"] = float(value)

        # Fill metrics
        organize_metrics(train_metrics_raw, final_metrics["train_metrics"])
        organize_metrics(val_metrics_raw, final_metrics["val_metrics"])
        organize_metrics(test_metrics_raw, final_metrics["test_metrics"])

        return final_metrics