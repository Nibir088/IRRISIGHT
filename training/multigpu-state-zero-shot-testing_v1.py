import sys
sys.path.append('/project/biocomplexity/wyr6fx(Nibir)/IJCAI-25_Irrigation_Mapping/Pytorch-Lightening/KIIM')

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import wandb
from pathlib import Path
from datetime import datetime
import torch.nn as nn
import torch

from data.data_module import IrrigationDataModule
from models.KIIM_v3 import KIIM
# from models.KIIM import KIIM
from utils.callbacks import TimerCallback
from utils.train_config import save_experiment_config  # if needed

@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IJCAI-25_Irrigation_Mapping/Pytorch-Lightening/KIIM/config", 
            config_name="multigpu-state-testing-KIIM-zero-shot_v1", version_base="1.2")
def test_model(cfg: DictConfig) -> None:
    """
    Test-only script: loads a model from checkpoint and runs the test loop.
    """
    print(f"*****************current gpu: {cfg.train.devices}, Attn: {cfg.model.use_attention}, Projection: {cfg.model.use_projection} ********************")
    
    pl.seed_everything(cfg.train.seed)
    
    # Initialize data module
    data_module = IrrigationDataModule(cfg, merge_train_valid=False)
    print(data_module._get_dataloader_kwargs())
    
    # Load model from checkpoint. Make sure cfg contains a valid checkpoint path.
    # For example, you could have cfg.test.checkpoint_path defined.
    checkpoint_path = cfg.test.ckpt  # e.g., "checkpoints/my_model.ckpt"
    model = KIIM.load_from_checkpoint(checkpoint_path,strict=False, **cfg.model)
    
    model.use_projection = False
    
    # print(model.backbone.model.channel_attention.fusion_weight)
    # For multi-GPU testing, convert BatchNorm layers to SyncBatchNorm if needed
    if len(cfg.train.devices) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Create a directory for logs if needed
    save_dir = Path(cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # (Optional) Setup loggers if you want to record test metrics
    loggers = []
    if cfg.logging.use_wandb:
        wandb.login(key='8e1d37626e9869704f83820542625ae247b94c13')
        wandb_logger = WandbLogger(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            save_dir=cfg.logging.save_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        loggers.append(wandb_logger)
    
    csv_logger = CSVLogger(
        save_dir=save_dir / "csv_logs",
        name=cfg.logging.run_name
    )
    loggers.append(csv_logger)
    
    # Initialize callbacks (only those that are relevant for testing)
    callbacks = []
    timer_callback = TimerCallback()
    callbacks.append(timer_callback)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Define multi-GPU strategy if needed
    strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=True
        )
    if len(cfg.train.devices) > 1:
        # Using DDP with proper settings for testing
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
    
    # Initialize trainer for testing only
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        precision=cfg.train.precision,
        deterministic=cfg.train.get('deterministic', False),
        sync_batchnorm=len(cfg.train.devices) > 1,
        log_every_n_steps=25,
        # No need for max_epochs because we are not training
    )
    
    # (Optional) Save experiment config if you wish to keep a record
    save_experiment_config(cfg, data_module, model, trainer, save_dir)
    
    # Run testing. Since the model is already loaded from checkpoint, you can simply pass it.
    test_results = trainer.test(model=model, datamodule=data_module)
    
    # Print test results
    print("Test Results:", test_results)
    
    # If using wandb, finish the run
    if cfg.logging.use_wandb:
        wandb.finish()
    if hasattr(model, 'get_metrics'):
        final_metrics = model.get_metrics()
        metrics_path = save_dir / "final_metrics.yaml"
        OmegaConf.save(config=final_metrics, f=metrics_path)

if __name__ == "__main__":
    test_model()
