import sys
sys.path.append('/project/biocomplexity/wyr6fx(Nibir)/IJCAI-25_Irrigation_Mapping/Pytorch-Lightening/KIIM')

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
import wandb
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import Trial
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import yaml
from datetime import datetime
import torch.nn as nn
import torch
from data.data_module import IrrigationDataModule
from models.KIIM_v3 import KIIM
from utils.callbacks import *
from utils.train_config import *


@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IJCAI-25_Irrigation_Mapping/Pytorch-Lightening/KIIM/config", config_name="multigpu-state-testing-KIIM-wo-tuning_v3", version_base="1.2")
def train(cfg: DictConfig) -> None:
    """Main training function with optional hyperparameter tuning."""
    
    print(f"*****************current gpu: {cfg.train.devices} ********************")
    
    pl.seed_everything(cfg.train.seed)
    
    
    # print(OmegaConf.to_yaml(config))
    

    # Initialize data module
    data_module = IrrigationDataModule(cfg, merge_train_valid=False)
    
    print(data_module._get_dataloader_kwargs())
    # Initialize model
    model = KIIM(**cfg.model)
    print(model)
    # Convert BatchNorm to SyncBatchNorm for multi-GPU training
    if len(cfg.train.devices) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    
    
    
#     #     # Create save directory
    save_dir = Path(cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
#     # Initialize loggers
    loggers = []
    # wandb.login(key='6f89939522657327198f880a89e67fca1d8a0f12')
    wandb.login(key='8e1d37626e9869704f83820542625ae247b94c13')
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
    # tb_logger = TensorBoardLogger(
    #     save_dir=save_dir / "tensorboard",
    #     name=cfg.logging.run_name,
    #     default_hp_metric=False
    # )
    # loggers.append(tb_logger)
    
    # CSV Logger
    csv_logger = CSVLogger(
        save_dir=save_dir / "csv_logs",
        # name=f"{cfg.logging.run_name}_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        name=cfg.logging.run_name
    )
    loggers.append(csv_logger)
    
    # Initialize callbacks
    callbacks: List[pl.Callback] = []
    
    # Timer callback
    timer_callback = TimerCallback()
    callbacks.append(timer_callback)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{epoch}-{val_iou_macro:.3f}",
        monitor="val_iou_macro_irr",
        mode="max",
        save_top_k=5,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if cfg.train.early_stopping:
        early_stopping = EarlyStopping(
            monitor=cfg.train.monitor,
            mode="max",
            patience=cfg.train.patience,
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Multi-GPU strategy 
    
    strategy = DDPStrategy(find_unused_parameters=True)
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
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        precision=cfg.train.precision,
        deterministic=cfg.train.get('deterministic', False),
        # Additional multi-GPU settings
        sync_batchnorm=len(cfg.train.devices) > 1,
        log_every_n_steps=25
    )
    
    # Track start time
    start_time = time.time()
    
    try:
        #####commenr this>>>
        # Train model
        # data_module.setup('test')
        # data_module.val_dataset = data_module.test_dataset['WA']
        trainer.fit(model, data_module)
        # print(trainer.callbacks)
        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        
        print("\nStarting testing phase...")
        
        # print(f"\nTest Results: {test_results}")

        # # Save the test results
        # test_results_path = save_dir / "test_results.yaml"
        # with open(test_results_path, "w") as f:
        #     yaml.safe_dump(test_results, f, default_flow_style=False)
        # print(f"Test results saved to {test_results_path}")


        
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
        # Save experiment config and close wandb
        save_experiment_config(cfg, data_module, model, trainer, save_dir)
        if cfg.logging.use_wandb:
            wandb.finish()
            
        current_logger = trainer.logger
        trainer.logger = None  # Disable the logger for testing
        trainer.test(ckpt_path="best", datamodule=data_module)
        
        # Save final metrics
        if hasattr(model, 'get_metrics'):
            final_metrics = model.get_metrics()
            metrics_path = save_dir / "final_metrics.yaml"
            OmegaConf.save(config=final_metrics, f=metrics_path)


if __name__ == "__main__":
    train()

