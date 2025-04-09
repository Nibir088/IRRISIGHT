import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from models.KIIM import KIIM
from data.data_module import IrrigationDataModule
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTester:
    def __init__(self, cfg: DictConfig, checkpoint_path: str):
        """
        Initialize the model tester.
        
        Args:
            cfg: Configuration object
            checkpoint_path: Path to the model checkpoint
        """
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize data module
        self.data_module = IrrigationDataModule(cfg.config_dir)
        self.data_module.setup('test')
        
        # Create save directory for results
        self.save_dir = Path(cfg.test.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> pl.LightningModule:
        """Load the trained model from checkpoint."""
        model = KIIM.load_from_checkpoint(
            self.checkpoint_path,
            **self.cfg.model
        )
        return model
    
    def test(self):
        """Run comprehensive testing on the model."""
        # Set up device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # Initialize metrics storage
        all_preds = []
        all_targets = []
        img_metrics = []
        
        # Test loop
        test_loader = self.data_module.test_dataloader()
        print("Starting model testing...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch['image'].to(device)
                targets = batch['mask'].to(device)
                
                # Forward pass
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Store predictions and targets
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
                # Calculate per-image metrics
                for i in range(len(images)):
                    metrics = self._calculate_image_metrics(
                        preds[i].cpu().numpy(),
                        targets[i].cpu().numpy()
                    )
                    img_metrics.append(metrics)
        
        # Calculate and save overall metrics
        self._save_metrics(all_preds, all_targets, img_metrics)
        
        # Generate and save visualizations
        self._generate_visualizations(all_preds, all_targets)
    
    def _calculate_image_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """Calculate metrics for a single image."""
        # Calculate IoU for each class
        ious = []
        for class_idx in range(self.cfg.model.num_classes):
            intersection = np.logical_and(pred == class_idx, target == class_idx).sum()
            union = np.logical_or(pred == class_idx, target == class_idx).sum()
            iou = intersection / union if union > 0 else 0
            ious.append(iou)
        
        # Calculate accuracy
        accuracy = (pred == target).mean()
        
        return {
            'iou_per_class': ious,
            'mean_iou': np.mean(ious),
            'accuracy': accuracy
        }
    
    def _save_metrics(self, all_preds: list, all_targets: list, img_metrics: list):
        """Save all computed metrics."""
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # Calculate classification report
        class_report = classification_report(
            all_targets,
            all_preds,
            output_dict=True
        )
        
        # Calculate average metrics across all images
        avg_metrics = {
            'mean_iou': np.mean([m['mean_iou'] for m in img_metrics]),
            'std_iou': np.std([m['mean_iou'] for m in img_metrics]),
            'mean_accuracy': np.mean([m['accuracy'] for m in img_metrics]),
            'std_accuracy': np.std([m['accuracy'] for m in img_metrics])
        }
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([avg_metrics])
        metrics_df.to_csv(self.save_dir / 'test_metrics.csv', index=False)
        
        # Save per-image metrics
        img_metrics_df = pd.DataFrame(img_metrics)
        img_metrics_df.to_csv(self.save_dir / 'per_image_metrics.csv', index=False)
        
        # Save classification report
        pd.DataFrame(class_report).transpose().to_csv(
            self.save_dir / 'classification_report.csv'
        )
        
        # Print summary
        print("\nTest Results:")
        print(f"Mean IoU: {avg_metrics['mean_iou']:.4f} ± {avg_metrics['std_iou']:.4f}")
        print(f"Mean Accuracy: {avg_metrics['mean_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}")
    
    def _generate_visualizations(self, all_preds: list, all_targets: list):
        """Generate and save visualization plots."""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        conf_matrix = confusion_matrix(all_targets, all_preds)
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close()
        
        # Plot IoU distribution
        plt.figure(figsize=(10, 6))
        iou_values = [m['mean_iou'] for m in img_metrics]
        plt.hist(iou_values, bins=50)
        plt.title('Distribution of IoU Scores')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.savefig(self.save_dir / 'iou_distribution.png')
        plt.close()

@hydra.main(config_path="../config", config_name="test")
def test(cfg: DictConfig) -> None:
    """Main testing function."""
    # Print test configuration
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    pl.seed_everything(cfg.test.seed)
    
    # Initialize and run tester
    tester = ModelTester(cfg, cfg.test.checkpoint_path)
    tester.test()

if __name__ == "__main__":
    test()