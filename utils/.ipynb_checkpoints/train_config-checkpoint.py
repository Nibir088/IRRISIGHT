import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import datetime
import torch

def save_experiment_config(
    cfg: DictConfig,
    data_module: pl.LightningDataModule,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    save_dir: Path
) -> None:
    """Save complete experiment configuration including model, data, and training setup."""
    
    # Create a comprehensive config dictionary
    full_config = {
        "experiment": {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "save_dir": str(save_dir)
        },
        "model": {
            "name": model.__class__.__name__,
            "hparams": OmegaConf.to_container(cfg.model, resolve=True),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        },
        "data": {
            "module_name": data_module.__class__.__name__,
            "batch_size": cfg.train.batch_size if hasattr(cfg.train, 'batch_size') else None,
            "num_workers": cfg.train.num_workers if hasattr(cfg.train, 'num_workers') else None,
            "dataset_config": {
                # "config_dir": str(cfg.config_dir),
                "data_dir": str(cfg.data.data_dir) if hasattr(cfg, 'data') and hasattr(cfg.data, 'data_dir') else None,
                "train_split": cfg.data.train_split if hasattr(cfg, 'data') and hasattr(cfg.data, 'train_split') else None,
                "val_split": cfg.data.val_split if hasattr(cfg, 'data') and hasattr(cfg.data, 'val_split') else None,
                "test_split": cfg.data.test_split if hasattr(cfg, 'data') and hasattr(cfg.data, 'test_split') else None
            }
        },
        "training": {
            "seed": cfg.train.seed,
            "max_epochs": trainer.max_epochs,
            "precision": trainer.precision,
            "accelerator": trainer.accelerator.value if hasattr(trainer.accelerator, 'value') else str(trainer.accelerator),
            "devices": trainer.device_ids if hasattr(trainer, 'device_ids') else trainer.devices,
            "strategy": str(trainer.strategy),
            "gradient_clip_val": trainer.gradient_clip_val,
            "accumulate_grad_batches": trainer.accumulate_grad_batches,
            "early_stopping": cfg.train.early_stopping if hasattr(cfg.train, 'early_stopping') else None,
            "patience": cfg.train.patience if hasattr(cfg.train, 'patience') else None
        },
        "optimization": {
            "optimizer": cfg.model.get('optimizer_name', 'AdamW'),
            "learning_rate": cfg.model.get('learning_rate', None),
            "weight_decay": cfg.model.get('weight_decay', None)
        }
    }
    
    # Save configurations
    config_dir = save_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    
    # Save full experiment config
    OmegaConf.save(config=full_config, f=config_dir / "experiment_config.yaml")
    
    # Save original Hydra config
    OmegaConf.save(config=cfg, f=config_dir / "hydra_config.yaml")
    
    # Save dataset information
    dataset_info = {}
    if hasattr(data_module, 'train_dataset') and data_module.train_dataset:
        dataset_info['train_size'] = len(data_module.train_dataset)
    if hasattr(data_module, 'val_dataset') and data_module.val_dataset:
        dataset_info['val_size'] = len(data_module.val_dataset)
    if hasattr(data_module, 'test_dataset') and data_module.test_dataset:
        dataset_info['test_size'] = len(data_module.test_dataset)
        
    OmegaConf.save(config=dataset_info, f=config_dir / "dataset_info.yaml")
