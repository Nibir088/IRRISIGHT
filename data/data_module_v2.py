

from typing import Optional, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data.dataset_v2 import ImageMaskDataset
import json
import yaml
    

class IrrigationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for irrigation dataset.

    This version assumes that the configuration is provided directly as a dictionary.
    """

    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Initialize the DataModule.

        Args:
            config: Configuration dictionary containing dataset and dataloader parameters.
            merge_train_valid: If True, merges train and validation datasets into one.
        """
        super().__init__()
        self.config = config

        # Extract configuration parameters
        self.dataset_params = self.config.get('dataset', {})
        self.dataloader_params = self.config.get('dataloader', {})
        




        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage: Either 'fit', 'test', or None
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageMaskDataset(
                data_dir = self.dataset_params.get('data_dir', ''),
                states = self.dataset_params.get('states', []),
                image_shape = self.dataset_params.get('image_shape', (224, 224)),
                transform = self.dataset_params.get('transform', False),
                gamma_value = self.dataset_params.get('gamma_value', 1.3),
                label_type = self.dataset_params.get('label_type', 'irrigation'),
                vision_indices = self.dataset_params.get('vision_indices', ['image']),
                train_type = self.dataset_params.get('train_type', 'cross-state'), 
                 split = 'train'
            )
            
            self.val_dataset = ImageMaskDataset(
                data_dir = self.dataset_params.get('data_dir', ''),
                states = self.dataset_params.get('states', []),
                image_shape = self.dataset_params.get('image_size', (224, 224)),
                transform = self.dataset_params.get('transform', False),
                gamma_value = self.dataset_params.get('gamma_value', 1.3),
                label_type = self.dataset_params.get('label_type', 'irrigation'),
                vision_indices = self.dataset_params.get('vision_indices', ['image']),
                train_type = self.dataset_params.get('train_type', 'cross-state'), 
                 split = 'val'
            )
            
        
        if (stage == 'test' or stage is None):
            
            self.test_dataset = ImageMaskDataset(
                data_dir = self.dataset_params.get('data_dir', ''),
                states = self.dataset_params.get('states', []),
                image_shape = self.dataset_params.get('image_size', (224, 224)),
                transform = self.dataset_params.get('transform', False),
                gamma_value = self.dataset_params.get('gamma_value', 1.3),
                label_type = self.dataset_params.get('label_type', 'irrigation'),
                vision_indices = self.dataset_params.get('vision_indices', ['image']),
                train_type = self.dataset_params.get('train_type', 'cross-state'), 
                 split = 'test'
            )
            

    def _get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for DataLoader from config."""
        return {
            'batch_size': self.dataloader_params.get('batch_size', 32),
            'num_workers': self.dataloader_params.get('num_workers', 4),
            'pin_memory': self.dataloader_params.get('pin_memory', True),
            'shuffle': False,
        }

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        kwargs = self._get_dataloader_kwargs()
        kwargs['shuffle'] = False  # Enable shuffling for training
        kwargs['drop_last'] = True
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs())

    def test_dataloader(self) -> Dict[str, DataLoader]:
        """Return the test DataLoaders for each state."""
        return DataLoader(self.test_dataset, **self._get_dataloader_kwargs())
#         if not self.test_dataset:
#             raise ValueError("Test datasets are not initialized or are empty.")

#         return {
#             state: DataLoader(dataset, **self._get_dataloader_kwargs())
#             for state, dataset in self.test_dataset.items()
#         }