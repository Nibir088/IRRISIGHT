import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
import wandb
from typing import Optional, Dict, Any, List
from pathlib import Path
from models.KIIM import KIIM
from data.data_module import IrrigationDataModule

def load_pretrained_model(checkpoint_path: str, model_config: DictConfig) -> KIIM:
    """
    Load a pretrained model with potential config modifications.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_config: New model configuration
    """
    # Load the model with new configs but keep weights
    model = KIIM(**model_config)
    
    # Load state dict from checkpoint
    checkpoint = pl.utilities.parsing.load_hparams_from_yaml(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

@hydra.main(config_path="config", config_name="train")
def train(cfg: DictConfig) -> None:
    """
    Main training function with fine-tuning support.
    
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
    data_module = IrrigationDataModule(cfg.config_dir)
    
    # Initialize model - with fine-tuning support
    if cfg.get('finetune') and cfg.finetune.get('checkpoint_path'):
        print(f"Loading pretrained model from {cfg.finetune.checkpoint_path}")
        model = KIIM.load_from_checkpoint(
            cfg.finetune.checkpoint_path,
            strict=cfg.finetune.get('strict', False),  # Allow loading with missing/extra keys
            **cfg.model
        )
        
        # Optionally freeze specific layers
        if cfg.finetune.get('freeze_backbone', False):
            for param in model.backbone.parameters():
                param.requires_grad = False
                
        if cfg.finetune.get('freeze_encoder', False):
            if hasattr(model.backbone, 'encoder'):
                for param in model.backbone.encoder.parameters():
                    param.requires_grad = False
        
        # Optionally modify learning rate for fine-tuning
        if cfg.finetune.get('learning_rate'):
            model.hparams.learning_rate = cfg.finetune.learning_rate
            
    else:
        print("Training model from scratch")
        model = KIIM(**cfg.model)
    
    # Initialize callbacks
    callbacks: List[pl.Callback] = []
    
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
        precision=cfg.train.precision
    )
    
    try:
        # Train model
        trainer.fit(model, data_module)
        
        # Save final model
        if cfg.train.save_model:
            trainer.save_checkpoint(save_dir / "final_model.ckpt")
            
        # Save best model path
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Best model path: {best_model_path}")
            with open(save_dir / "best_model_path.txt", "w") as f:
                f.write(best_model_path)
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # Close wandb
        if cfg.logging.use_wandb:
            wandb.finish()
        
        # Save final metrics
        if hasattr(model, 'get_metrics'):
            final_metrics = model.get_metrics()
            metrics_path = save_dir / "final_metrics.yaml"
            OmegaConf.save(config=final_metrics, f=metrics_path)

if __name__ == "__main__":
    train()