

from typing import Optional, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from data.dataset import ImageMaskDataset
import json
import yaml
    

class IrrigationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for irrigation dataset.

    This version assumes that the configuration is provided directly as a dictionary.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        merge_train_valid: Optional[bool] = False,
    ):
        """
        Initialize the DataModule.

        Args:
            config: Configuration dictionary containing dataset and dataloader parameters.
            merge_train_valid: If True, merges train and validation datasets into one.
        """
        super().__init__()
        self.config = config
        self.merge = merge_train_valid

        # Extract configuration parameters
        self.dataset_params = self.config.get('dataset', {})
        self.dataloader_params = self.config.get('dataloader', {})
        


        with open(self.dataset_params.get('data_file_base_name'), 'r') as file:
            self.data_file_paths = yaml.safe_load(file)  # Load YAML instead of JSON

        # self.data_file_paths = self.data_file[self.dataset_params.get('data_file_index')]
        # self.data_file_paths = self.data_file[self.dataset_params.get('data_file_index')]


        # self.data_file = self.dataset_params.get('data_file_base_name')
        # self.data_file_paths = self.dataset_params.get('data_file_index', {})

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
            patches = list(self.data_file_paths['train']['patches'].values())
            masks = list(self.data_file_paths['train']['masks'].values())

            train_len = int(len(patches) * self.dataset_params.get('train_size', 20) /100.0)
            

            # train_len = len(self.data_file_paths['train']['patches'])
            self.train_dataset = ImageMaskDataset(
                image_paths = patches[:train_len],
                mask_paths = masks[:train_len],
                states=self.dataset_params.get('states', []),
                image_size=self.dataset_params.get('image_size', (256, 256)),
                transform=self.dataset_params.get('transform', False),
                gamma_value=self.dataset_params.get('gamma_value', 1.3),
                is_binary=self.dataset_params.get('is_binary', False),
                image_types=self.dataset_params.get('image_types', ['image']),
                agri_indices=self.dataset_params.get('agri_indices', []),
                source = self.dataset_params.get('source', 'landsat')
            )
            self.val_dataset = ImageMaskDataset(
                    image_paths = patches[train_len:],
                    mask_paths = masks[train_len:],
                    states=self.dataset_params.get('states', []),
                    image_size=self.dataset_params.get('image_size', (256, 256)),
                    transform=self.dataset_params.get('transform', False),
                    gamma_value=self.dataset_params.get('gamma_value', 1.3),
                    is_binary=self.dataset_params.get('is_binary', False),
                    image_types=self.dataset_params.get('image_types', ['image']),
                    agri_indices=self.dataset_params.get('agri_indices', []),
                    source = self.dataset_params.get('source', 'landsat')
            )
                
            # if self.merge:
            #     self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
        
        if (stage == 'test' or stage is None):
            
            self.test_dataset = {}
            states = self.dataset_params.get('states', [])
            # states = ['FL','UT']
            for state in states:
                data_file_paths = self.data_file[state]
                test_data = ImageMaskDataset(
                    image_paths = data_file_paths['test']['patches'],
                    mask_paths = data_file_paths['test']['masks'],
                    states=states,
                    image_size=self.dataset_params.get('image_size', (256, 256)),
                    transform=self.dataset_params.get('transform', False),
                    gamma_value=self.dataset_params.get('gamma_value', 1.3),
                    is_binary=self.dataset_params.get('is_binary', False),
                    image_types=self.dataset_params.get('image_types', ['image']),
                    agri_indices=self.dataset_params.get('agri_indices', []),
                    source=self.dataset_params.get('source', 'landsat')
                )
                self.test_dataset[state] = test_data

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
        if not self.test_dataset:
            raise ValueError("Test datasets are not initialized or are empty.")

        return {
            state: DataLoader(dataset, **self._get_dataloader_kwargs())
            for state, dataset in self.test_dataset.items()
        }