import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pathlib import Path
import time, yaml
import torch.nn as nn
import wandb
import pandas as pd
from data.data_module_v2 import IrrigationDataModule
from model_v3.TeacherModel import TeacherModel
from utils.train_config import save_experiment_config
import torch

torch.set_float32_matmul_precision('high')
def find_dir(train_type, test_only, use_pretrained, use_attn, use_proj, use_vlm, vlm_type, freeze, use_text, use_rgb, use_land, use_crop, use_veg, states,pretrained_model):
    out = "/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_Irrigation_Mapping_Model/Output_v2/"
    if train_type == 'cross-state':
        out += 'cross/'
    elif train_type == 'holdout':
        out += 'holdout/'  # Or receive states as input if needed
        for state,pct in states:
            out+=f'{state}{pct}_'
        out+='/'
        

    out += 'zero_shot/' if test_only else 'trained/'

    if use_pretrained:
        out += 'pretrained_'
    if use_attn:
        out += 'attnmask_'
    if use_proj:
        out += 'projcrop_'
    if use_vlm:
        out += 'vlm_'
    out += '/'

    if use_pretrained:
        out += pretrained_model
    elif use_vlm:
        out += vlm_type
        if freeze:
            out += '_freezed'
        if use_text:
            out += '_text'
    out += '/'

    if use_rgb:
        out += 'rgb_'
    if use_land:
        out += 'landmask_'
    if use_crop:
        out += 'cropmask_'
    if use_veg:
        out += 'vegetation_'

    return out
OmegaConf.register_new_resolver("find_dir", find_dir)

@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_Irrigation_Mapping_Model/KIIM/config", config_name="supervised_training_gpu_v2", version_base="1.2")
def train(cfg: DictConfig) -> None:
    print(f"Running on GPUs: {cfg.train.devices}")
    pl.seed_everything(cfg.train.seed)


    # print("Hydra output dir set to:", cfg.hydra.run.dir)

    data_module = IrrigationDataModule(cfg)
    data_module.setup('fit')
    data_module.setup('test')

    model = TeacherModel(**cfg)
    print(model)

    if isinstance(cfg.train.devices, (list, tuple)) and len(cfg.train.devices) > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    save_dir = Path(cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir / "checkpoints",
            filename="{epoch}-{val_iou_macro_irr:.3f}",
            monitor=cfg.train.monitor,
            mode="max",
            save_top_k=1,
            save_last=True,
        )
    ]

    if cfg.train.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor,
                mode="max",
                patience=cfg.train.patience,
                verbose=True
            )
        )

    trainer_kwargs = dict(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        precision=cfg.train.precision,
        deterministic=cfg.train.get('deterministic', False),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    if cfg.train.accelerator != "cpu" and len(cfg.train.devices) > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(**trainer_kwargs)

    if cfg.general.test_only:
        test_results = trainer.test(model, datamodule=data_module)
        test_df = pd.DataFrame(test_results)
        test_df.to_csv(save_dir / "test_results.csv", index=False)
    else:
        try:
            trainer.fit(model, data_module)

            best_model_path = callbacks[0].best_model_path
            print(f"Best model saved at: {best_model_path}")
            with open(save_dir / "best_model_path.txt", "w") as f:
                f.write(best_model_path)

            print("\nRunning validation and test...\n")

            val_results = trainer.validate(model, datamodule=data_module)
            val_df = pd.DataFrame(val_results)
            val_df.to_csv(save_dir / "validation_results.csv", index=False)

            test_results = trainer.test(model, datamodule=data_module)
            test_df = pd.DataFrame(test_results)
            test_df.to_csv(save_dir / "test_results.csv", index=False)

            if cfg.train.save_model:
                trainer.save_checkpoint(save_dir / "final_model.ckpt")

        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    save_experiment_config(cfg, data_module, model, trainer, save_dir)
    if hasattr(model, 'get_metrics'):
        final_metrics = model.get_metrics()
        OmegaConf.save(config=final_metrics, f=save_dir / "final_metrics.yaml")


if __name__ == "__main__":
    train()
