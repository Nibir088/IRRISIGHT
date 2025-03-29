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
from plotly.io import write_html, write_image
from optuna.terminator import report_cross_validation_scores
from optuna.visualization import plot_terminator_improvement
from data.data_module import IrrigationDataModule
from utils.callbacks import *
from utils.train_config import *
from models.KIIM import KIIM

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
                trial_cfg.hparam_tuning.search_space.learning_rate  # Specify the discrete values you want to allow 5e-4
            )
        }
        
        if not trial_cfg.model.loss_config.use_ce:
            # Suggest weights uniformly and ensure sum <= 1
            focal_weight = trial.suggest_float('focal_weight', 0.2, 1.0, step=0.05)
            kg_weight = trial.suggest_float('kg_weight', 0.0, min(0.25, 1.0-focal_weight), step=0.05)
            dice_weight = trial.suggest_float('dice_weight', 0.0, min(0.4, 1.0-focal_weight-kg_weight), step=0.05)
                                            
            stream_weight = 1.0-kg_weight-dice_weight-focal_weight
            
            model_params['loss_config'] = {
                "ce_weight": 0.0,
                "kg_weight": kg_weight,
                "dice_weight": dice_weight,
                "focal_weight": focal_weight,
                "stream_weight": stream_weight
            }
        
        dataloader_param =  {
                'batch_size': trial.suggest_categorical(
                    'batch_size',
                    trial_cfg.hparam_tuning.search_space.batch_size
                )
            }
        
        print(model_params)
        
        trial_cfg.dataloader.update(dataloader_param)
        trial_cfg.model.update(model_params)
        
        # Create trial-specific save directory
        trial_save_dir = self.base_save_dir / f"trial_{trial.number}"
        trial_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data module
        data_module = IrrigationDataModule(trial_cfg)
        print(data_module._get_dataloader_kwargs())
        # print(data_module.data_file_paths)
        print(data_module.data_file_paths['train']['len'], len(data_module.data_file_paths['train']['image']))
        # Initialize model with trial parameters
        model = KIIM(**trial_cfg.model)
        # print(model, trial_cfg.model)
        # print(model_params['loss_config'])
        # Create logger for this trial
        logger = TensorBoardLogger(
            save_dir=str(trial_save_dir),
            name="trial_logs",
            version=str(trial.number)
        )
        
        # Create callbacks
        callbacks = self.create_callbacks(trial, trial_save_dir)
        
        # Multi-GPU strategy setup if needed
        # strategy = 'ddp'
        strategy=DDPStrategy(find_unused_parameters=True)
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
            best_score = trainer.callback_metrics.get(trial_cfg.train.monitor, torch.tensor(0.0))
            print(trainer.callback_metrics)
            return best_score.item()
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return float('-inf')  # Return worst possible score on failure

def run_hyperparameter_optimization(cfg: DictConfig) -> Dict[str, Any]:
    """Run hyperparameter optimization using Optuna."""
    
    # Create base directory for all trials
    base_save_dir = Path("hparam_tuning")
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
    
    #####save figure.....
    fig1 = optuna.visualization.plot_intermediate_values(study)
    fig2 = optuna.visualization.plot_optimization_history(study)
    # fig3 = plot_terminator_improvement(study, plot_error=True)

    # Save as interactive HTML files
    write_html(fig1, str(results_dir / "intermediate_values.html"))
    write_html(fig2, str(results_dir / "optimization_history.html"))
    # write_html(fig3, str(results_dir / "terminator_improvement.html"))

    # Optional: Save as static images (PNG format)
    write_image(fig1, str(results_dir / "intermediate_values.png"))
    write_image(fig2, str(results_dir / "optimization_history.png"))
    # write_image(fig3, str(results_dir / "terminator_improvement.png"))
    #####################
    
    # Save best parameters and results
    best_params = study.best_params
    
    print('best params from here: ', best_params)
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
    print('save directory: ', results_dir)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value: {best_value:.4f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    
    
    return best_params



@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IJCAI-25_Irrigation_Mapping/Pytorch-Lightening/KIIM/config", config_name="multigpu-state-training-baseline-benchmark_v2", version_base="1.2")
def train(cfg: DictConfig) -> None:
    """Main training function with optional hyperparameter tuning."""
    
    print(f"*****************current gpu: {cfg.train.devices} ********************")
    
    pl.seed_everything(cfg.train.seed)

    override_param = {}
    if cfg.get('hparam_tuning', {}).get('enabled', False):
        print("\nStarting hyperparameter optimization...")
        best_params = run_hyperparameter_optimization(cfg)
        # print(best_params)
        # Update configuration with best parameters
        
        
        cfg.model.learning_rate = best_params['learning_rate']
        if not cfg.model.loss_config.use_ce:
            cfg.model.loss_config = {
                "ce_weight": 0.0,
                "dice_weight": best_params['dice_weight'],
                "focal_weight": best_params['focal_weight'],
                "kg_weight": best_params['kg_weight'],
                "stream_weight": 1.0-best_params['dice_weight']-best_params['focal_weight']-best_params['kg_weight']
            }
        
        cfg.dataloader.batch_size = best_params['batch_size']
        cfg.train.batch_size = best_params['batch_size']
        
#         # cfg.model.update(best_params)
        print("\nUpdated configuration with best parameters. Starting training...")
    
    
    # Initialize data module
    data_module = IrrigationDataModule(cfg, merge_train_valid=True)
    
    # Combine training and validation datasets
#     data_module.setup('fit')
#     data_module.setup('test')
    
#     combined_dataset = torch.utils.data.ConcatDataset([data_module.train_dataset, data_module.val_dataset])
#     data_module.train_dataset = combined_dataset  # Update the train dataset
    print(data_module._get_dataloader_kwargs())
    # Initialize model
    model = KIIM(**cfg.model)
    
    # Convert BatchNorm to SyncBatchNorm for multi-GPU training
    if len(cfg.train.devices) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    
    
    
    #     # Create save directory
    save_dir = Path(cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    
#     # Initialize loggers
    loggers = []
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
    
    # Initialize callbacks
    callbacks: List[pl.Callback] = []
    
    # Timer callback
    timer_callback = TimerCallback()
    callbacks.append(timer_callback)
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / "checkpoints",
        filename="{epoch}-{val_iou_macro_irr:.3f}",
        monitor=cfg.train.monitor,
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
    
    # strategy = 'ddp'
    strategy = DDPStrategy(
            find_unused_parameters=True
        )
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
        # Train model
        trainer.fit(model, data_module)
        
        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # current_logger = trainer.logger
        # trainer.logger = None  # Disable the logger for testing
        print("\nStarting testing phase...")
        
        # trainer.test(ckpt_path="best", datamodule=data_module)
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

