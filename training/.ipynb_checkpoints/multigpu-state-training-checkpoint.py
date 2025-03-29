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
from models.KIIM import KIIM
from utils.callbacks import *
from utils.train_config import *

class OptunaObjective:
    """Objective class for Optuna hyperparameter optimization."""
    def __init__(self, cfg: DictConfig, base_save_dir: Path):
        self.cfg = cfg
        self.base_save_dir = base_save_dir
        self.callbacks = None
    def create_callbacks(self, trial: Trial, save_dir: Path) -> List[pl.Callback]:
        """Create callbacks for the trial."""
        self.callbacks = CustomCallbackManager(save_dir=save_dir, trial=trial)
        self.callbacks.add_timer_callback()
        self.callbacks.add_early_stopping(mode=self.cfg.train.mode,monitor=self.cfg.train.monitor, patience=self.cfg.train.patience, verbose=self.cfg.train.verbose )
        self.callbacks.add_learning_rate_monitor(logging_interval='step')
        self.callbacks.add_pruning(monitor=self.cfg.train.monitor)
        self.callbacks.add_model_checkpoint(mode=self.cfg.train.mode, monitor=self.cfg.train.monitor, save_top_k=self.cfg.train.top_k, verbose=self.cfg.train.verbose)
        self.callbacks.add_clear_memory_callback()
    def __call__(self, trial: Trial) -> float:
        # Define hyperparameter search space
        
        
        # Update configuration with trial parameters
        trial_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))


        model_params = {
            'learning_rate': trial.suggest_categorical(
                'learning_rate',
                [1e-3,  2e-4, 1e-4]  # Specify the discrete values you want to allow 5e-4
            )
        }
        
        if not trial_cfg.model.loss_config.use_ce:
            kg_weight = trial.suggest_float('kg_weight', 0.1, 0.3, step=0.2)
            dice_weight = trial.suggest_float('dice_weight', 0.1, 0.3, step=0.2)
            focal_weight = trial.suggest_float('focal_weight', 0.1, 0.5, step=0.2)
            stream_weight = trial.suggest_float('stream_loss', 0.1,0.3, step=0.2)

            scale = kg_weight+dice_weight+focal_weight+stream_weight

            # Add loss config to model parameters
            model_params['loss_config'] = {
                "ce_weight": 0.0,
                "dice_weight": float(format(dice_weight/scale, '.3f')),
                "focal_weight": float(format(focal_weight/scale, '.3f')),
                "kg_weight": float(format(kg_weight/scale, '.3f')),
                "stream_weight": float(format(stream_weight/scale, '.3f'))
            }
        
        override_param = {
            'dataloader': {
                'batch_size': trial.suggest_int(
                    'batch_size',
                    16,  # min value
                    32,  # max value
                    step=16  # step size
                )
            }
        }
    

        trial_cfg.model.update(model_params)
        # Create trial-specific save directory
        trial_save_dir = self.base_save_dir / f"trial_{trial.number}"
        trial_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data module
        data_module = IrrigationDataModule(trial_cfg.config_dir, override_params=override_param)
        
        # Initialize model with trial parameters
        model = KIIM(**trial_cfg.model)
        
        # Create logger for this trial
        logger = TensorBoardLogger(
            save_dir=str(trial_save_dir),
            name="trial_logs",
            version=str(trial.number)
        )
        
        # Create callbacks
        callbacks = self.create_callbacks(trial, trial_save_dir)
        
        # Multi-GPU strategy setup if needed
        strategy = 'ddp'
        # strategy = DDPStrategy(
        #     find_unused_parameters=False,
        #     gradient_as_bucket_view=True,
        #     static_graph=True,
        #     process_group_backend='nccl',
        #     bucket_cap_mb=25
        # )
        # if len(trial_cfg.train.devices) > 1:
        #     strategy = pl.strategies.DDPStrategy(
        #         find_unused_parameters=False,
        #         gradient_as_bucket_view=True
        #     )
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=trial_cfg.train.max_epochs,
            accelerator=trial_cfg.train.accelerator,
            devices=trial_cfg.train.devices,
            strategy=strategy,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=trial_cfg.train.gradient_clip_val,
            precision=trial_cfg.train.precision,
            deterministic=trial_cfg.train.get('deterministic', False),
            enable_progress_bar=True,
            enable_model_summary=True,
            sync_batchnorm=len(trial_cfg.train.devices) > 1,
            log_every_n_steps=5,
        )
        
        try:
            # Train with this trial's parameters
            trainer.fit(model, data_module)
            
            # Get the best validation score
            best_score = trainer.callback_metrics.get("val_iou_macro", torch.tensor(0.0))
            print(trainer.callback_metrics)
            return best_score.item()
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf')  # Return worst possible score on failure



def run_hyperparameter_optimization(cfg: DictConfig) -> Dict[str, Any]:
    """Run hyperparameter optimization using Optuna."""
    
    # Create base directory for all trials
    base_save_dir = Path(cfg.train.save_dir) / "hparam_tuning"
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{cfg.logging.project_name}_hparam_tuning",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=10
        )
    )
    
    # Run optimization
    optimization_fnc = OptunaObjective(cfg, base_save_dir)
    study.optimize(
        optimization_fnc,
        n_trials=cfg.hparam_tuning.n_trials,
        timeout=cfg.hparam_tuning.timeout_hours * 3600 if hasattr(cfg.hparam_tuning, 'timeout_hours') else None,
        gc_after_trial=True,
        # callbacks=PyTorchLightningPruningCallback(
        #     trial=self.trial,
        #     monitor=monitor
        # ),
        show_progress_bar=True
    )
    
    # Save study results
    results_dir = base_save_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save best parameters and results
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "best_params": best_params,
        "best_value": float(best_value),
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "direction": study.direction.name
    }
    
    # Save results to YAML
    with open(results_dir / "optimization_results.yaml", "w") as f:
        yaml.safe_dump(results, f, default_flow_style=False)
    
    # Save study statistics
    stats = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": float(study.best_trial.value),
            "params": study.best_trial.params
        },
        "n_trials": len(study.trials),
        "n_complete_trials": len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE])),
        "n_pruned_trials": len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])),
        "n_failed_trials": len(study.get_trials(states=[optuna.trial.TrialState.FAIL]))
    }
    
    # Save statistics to YAML
    with open(results_dir / "study_statistics.yaml", "w") as f:
        yaml.safe_dump(stats, f, default_flow_style=False)
    
    print("\nHyperparameter Optimization Results:")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {best_value:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params


@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IJCAI-25_Irrigation_Mapping/Pytorch-Lightening/KIIM/config", config_name="multigpu-state-training-baseline")
def train(cfg: DictConfig) -> None:
    """Main training function with optional hyperparameter tuning."""
    
    # Print config
    # print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.train.seed)
    # cfg.hydra.run.dir = f"outputs/{cfg.dataset.data_file_index}"
    # Run hyperparameter tuning if enabled
    if cfg.get('hparam_tuning', {}).get('enabled', False):
        print("\nStarting hyperparameter optimization...")
        best_params = run_hyperparameter_optimization(cfg)
        print(best_params)
        # Update configuration with best parameters
        
        
        cfg.model.learning_rate = best_params['learning_rate']
        if not cfg.model.loss_config.use_ce:
            cfg.model.loss_config = {
                "ce_weight": 0.0,
                "dice_weight": best_params['dice_weight'],
                "focal_weight": best_params['focal_weight'],
                "kg_weight": best_params['kg_weight'],
                "stream_weight": best_params['stream_loss']
            }
        cfg.dataloader.batch_size = best_params['batch_size']
        
        # cfg.model.update(best_params)
        print("\nUpdated configuration with best parameters. Starting training...")
    
    # Create save directory
    save_dir = Path(cfg.train.save_dir) / cfg.logging.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loggers
    loggers = []
    wandb.login(key='6f89939522657327198f880a89e67fca1d8a0f12')
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
        # name=f"{cfg.logging.run_name}_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        name=cfg.logging.run_name
    )
    loggers.append(csv_logger)
    
    # Initialize data module
    data_module = IrrigationDataModule(cfg.config_dir)
    
    # Initialize model
    model = KIIM(**cfg.model)
    
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
    strategy = 'ddp'
    # if len(cfg.train.devices) > 1:
    #     strategy = DDPStrategy(
    #         find_unused_parameters=False,
    #         gradient_as_bucket_view=True
    #     )
    
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

