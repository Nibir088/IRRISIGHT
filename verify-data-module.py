import pytorch_lightning as pl
from data.data_module import *
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path

def visualize_batch(batch, num_samples=4):
    """Visualize a batch of data."""
    if not batch:  # Check if batch is None or empty
        print("No data to visualize")
        return
        
    try:
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*4, 8))
        
        # Plot images and masks
        for i in range(min(num_samples, batch['image'].shape[0])):  # Ensure we don't exceed batch size
            if 'image' in batch:
                # Plot image
                img = batch['image'][i].permute(1, 2, 0).numpy()  # Convert to HWC format
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Image {i}')
                axes[0, i].axis('off')
            
            if 'true_mask' in batch:
                # Plot mask
                mask = batch['true_mask'][i].squeeze().numpy()
                axes[1, i].imshow(mask, cmap='viridis')
                axes[1, i].set_title(f'Mask {i}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error visualizing batch: {str(e)}")

def get_batch_safely(loader):
    """Safely get a batch from a loader."""
    if loader is None:
        return None
        
    try:
        return next(iter(loader))
    except StopIteration:
        print("Loader is empty")
        return None
    except Exception as e:
        print(f"Error loading batch: {str(e)}")
        return None

def print_dataset_info(name, dataset):
    """Print information about a dataset if it exists."""
    if dataset is not None:
        print(f"{name} dataset: {len(dataset)} samples")
    else:
        print(f"{name} dataset: None")

def inspect_dataset(datamodule):
    """Print information about the dataset and visualize samples."""
    print("Setting up data...")
    try:
        datamodule.setup('fit')
    except Exception as e:
        print(f"Error setting up fit data: {str(e)}")
        
    try:
        datamodule.setup('test')
    except Exception as e:
        print(f"Error setting up test data: {str(e)}")
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print_dataset_info("Train", datamodule.train_dataset)
    print_dataset_info("Validation", datamodule.val_dataset)
    print_dataset_info("Test", datamodule.test_dataset)
    
    # Get data loaders safely
    train_loader = datamodule.train_dataloader() if hasattr(datamodule, 'train_dataloader') else None
    val_loader = datamodule.val_dataloader() if hasattr(datamodule, 'val_dataloader') else None
    test_loader = datamodule.test_dataloader() if hasattr(datamodule, 'test_dataloader') else None
    
    # Print batch information for training data
    print("\nBatch information:")
    train_batch = get_batch_safely(train_loader)
    
    if train_batch:
        print("\nKeys in batch:", train_batch.keys())
        for key, value in train_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape:", value.shape)
                print(f"{key} dtype:", value.dtype)
    else:
        print("No training batch available")
    
    # Visualize samples from each split
    print("\nVisualizing training samples...")
    if train_batch:
        visualize_batch(train_batch)
    else:
        print("No training samples to visualize")
    
    print("\nVisualizing validation samples...")
    val_batch = get_batch_safely(val_loader)
    if val_batch:
        visualize_batch(val_batch)
    else:
        print("No validation samples to visualize")
    
    print("\nVisualizing test samples...")
    test_batch = get_batch_safely(test_loader)
    if test_batch:
        visualize_batch(test_batch)
    else:
        print("No test samples to visualize")

def main():
    # Load config
    config_path = "config/train.yaml"  # Update this path
    print(f"Loading config from {config_path}")
    
    try:
        # Create datamodule
        datamodule = IrrigationDataModule(config_path)
        
        # Inspect the dataset
        inspect_dataset(datamodule)
        
        # Test a full epoch only if training dataset exists
        if hasattr(datamodule, 'train_dataloader'):
            print("\nTesting full epoch iteration...")
            train_loader = datamodule.train_dataloader()
            if train_loader is not None:
                for i, batch in enumerate(train_loader):
                    if i % 10 == 0:  # Print progress every 10 batches
                        print(f"Successfully loaded batch {i}")
                print("Completed full epoch iteration")
            else:
                print("No training loader available")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set random seed for reproducibility
    pl.seed_everything(42)
    main()