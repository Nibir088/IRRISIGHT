import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Timer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
import wandb
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
from datetime import datetime, timedelta
import torch.nn as nn
import torch
from data.data_module import *
from models.KIIM import *
class TimerCallback(pl.Callback):
    """Custom callback to track training time."""
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
    
    def on_train_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        trainer.logger.log_metrics({
            'total_training_hours': hours,
            'total_training_time': f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        })
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        trainer.logger.log_metrics({
            'epoch_time_seconds': epoch_time,
            'epoch_time': f"{epoch_time:.2f}s"
        })
def save_experiment_config(
    cfg: DictConfig,
    data_module: pl.LightningDataModule,
    model: pl.LightningModule,
    trainer: pl.Trainer,
    save_dir: Path
) -> None:
    """
    Save complete experiment configuration including model, data, and training setup.
    
    Args:
        cfg (DictConfig): Original Hydra configuration
        data_module (pl.LightningDataModule): Data module instance
        model (pl.LightningModule): Model instance
        trainer (pl.Trainer): Trainer instance
        save_dir (Path): Directory to save configurations
    """
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
                "config_dir": str(cfg.config_dir),
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
            # "deterministic": trainer.deterministic,
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
    OmegaConf.save(
        config=full_config,
        f=config_dir / "experiment_config.yaml"
    )
    
    # Save original Hydra config
    OmegaConf.save(
        config=cfg,
        f=config_dir / "hydra_config.yaml"
    )
    
    # Save model hparams separately for easy access
    # if hasattr(model, 'hparams'):
    #     OmegaConf.save(
    #         config=OmegaConf.create(model.hparams),
    #         f=config_dir / "model_hparams.yaml"
    #     )
    
    # Save dataset information
    dataset_info = {}
    if hasattr(data_module, 'train_dataset') and data_module.train_dataset:
        dataset_info['train_size'] = len(data_module.train_dataset)
    if hasattr(data_module, 'val_dataset') and data_module.val_dataset:
        dataset_info['val_size'] = len(data_module.val_dataset)
    if hasattr(data_module, 'test_dataset') and data_module.test_dataset:
        dataset_info['test_size'] = len(data_module.test_dataset)
        
    OmegaConf.save(
        config=dataset_info,
        f=config_dir / "dataset_info.yaml"
    )
@hydra.main(config_path="../config", config_name="train-multi-gpu")
def train(cfg: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        cfg (DictConfig): Hydra configuration
    """
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.train.seed)
    
    # Create save directory
    save_dir = Path(cfg.train.save_dir) / cfg.logging.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loggers
    loggers = []
    
    # WandB Logger
    if cfg.logging.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            save_dir=cfg.logging.save_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
    
    # TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir=save_dir / "tensorboard",
        name=cfg.logging.run_name,
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # CSV Logger
    csv_logger = CSVLogger(
        save_dir=save_dir / "csv_logs",
        name=cfg.logging.run_name
    )
    loggers.append(csv_logger)
    
    # Initialize data module
    data_module = IrrigationDataModule(
        cfg.config_dir
    )
    
    # Initialize model
    model = KIIM(
        **cfg.model
    )
    
    # Convert BatchNorm to SyncBatchNorm for multi-GPU training
    if len(cfg.train.devices) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Initialize callbacks
    callbacks: List[pl.Callback] = []
    
    # Timer callback
    timer_callback = TimerCallback()
    callbacks.append(timer_callback)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{epoch}-{val_iou_macro:.3f}",
        monitor="val_iou_macro",
        mode="max",
        save_top_k=5,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if cfg.train.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_iou_macro",
            mode="max",
            patience=cfg.train.patience,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Multi-GPU strategy setup
    strategy = None
    if len(cfg.train.devices) > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        precision=cfg.train.precision,
        deterministic=cfg.train.get('deterministic', False),
        # Additional multi-GPU settings
        sync_batchnorm=len(cfg.train.devices) > 1,
        # replace_sampler_ddp=True
    )
    
    # Track start time
    start_time = time.time()
    
    try:
        # Train model
        trainer.fit(model, data_module)
        
        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        print(f"\nTotal training time: {training_time_str}")
        
        # Save final model
        if cfg.train.save_model:
            trainer.save_checkpoint(save_dir / "final_model.ckpt")
            
        # Save best model path
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Best model path: {best_model_path}")
            with open(save_dir / "best_model_path.txt", "w") as f:
                f.write(best_model_path)
        
        # Save training time
        with open(save_dir / "training_time.txt", "w") as f:
            f.write(f"Training time: {training_time_str}\n")
            f.write(f"Started: {datetime.fromtimestamp(start_time)}\n")
            f.write(f"Finished: {datetime.fromtimestamp(end_time)}\n")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # Close wandb
        save_experiment_config(cfg, data_module, model, trainer, save_dir)
        if cfg.logging.use_wandb:
            wandb.finish()
        
        # Save final metrics
        if hasattr(model, 'get_metrics'):
            final_metrics = model.get_metrics()
            metrics_path = save_dir / "final_metrics.yaml"
            OmegaConf.save(config=final_metrics, f=metrics_path)

if __name__ == "__main__":
    train()
    
    
