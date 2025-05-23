import sys
# sys.path.append('/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM')

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
# from models.TeacherStudentModel_v1 import TeacherStudentKIIM
from utils.train_config import save_experiment_config
import torch
torch.set_float32_matmul_precision('high')



@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_Irrigation_Mapping_Model/KIIM/config", config_name="supervised_training_gpu", version_base="1.2")
def train(cfg: DictConfig) -> None:
    print(f"Running on GPUs: {cfg.train.devices}")
    pl.seed_everything(cfg.train.seed)
    
    
    
    def find_dir():
        if cfg.dataset.train_type=='cross-state':
            out = 'cross'
        else:
            for state, value in cfg.dataset.states:
                out += f'{state}{value}'
                # print(f"State: {state}, Value: {value}")
        
        out+='/'
        if cfg.test_only:
            out += f'zero_shot/'
        else:
            out += 'trained/'
        temp_cfg = cfg.teachermodel.BaseModel
        if temp_cfg.use_pretrained_module:
            out += 'pretrained_'
        if temp_cfg.use_attention_module:
            out += 'attnmask_'
        if temp_cfg.use_projection_module:
            out += 'projcrop_'
        if temp_cfg.use_vlm_module:
            out += 'vlm_'
        out += '/'
        if temp_cfg.use_pretrained_module:
            temp_2_cfg = cfg.teachermodel.PretrainedModule.model_name
            out+= temp_2_cfg
        elif temp_cfg.use_vlm_module:
            temp_2_cfg = cfg.teachermodel.VLMModule
            out+= temp_2_cfg.vlm_type
            if temp_2_cfg.freeze_model:
                out+= '_freezed'
            if temp_2_cfg.use_text:
                out+= '_text'
        out+= '/'
        
        mlm_cfg = cfg.teachermodel.MultimodalImageryModule
        if mlm.use_rgb:
            out+= 'rgb_'
        if mlm.use_land_mask:
            out+= 'landmask_'
        if use_crop_mask:
            out+= 'cropmask_'
        if use_vegetation:
            out+= 'vegetation_'
        return out
        
    
    # Dynamically override hydra.run.dir
    cfg.hydra.run.dir += find_dir()

    # Optional: Print or log it
    print("Hydra output dir set to:", cfg.hydra.run.dir)

    
    

    data_module = IrrigationDataModule(cfg)
    data_module.setup('fit')
    data_module.setup('test')
    
#     print(data_module.train_dataloader().batch_size, len(data_module.train_dataloader()))
    
    
# #     print(data_module.train_dataset)
    # print(**cfg)
    # student_model = KIIM(**cfg.model)
    # print(student_model)
#     teacher_model = KIIM(**cfg.model)
    
#     # checkpoint = torch.load("/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/AZ/result_stats/checkpoints/epoch=45-val_iou_macro_irr=0.734.ckpt")
#     # teacher_model.load_state_dict(checkpoint['state_dict'], strict=False)

#     # teacher_model = TeacherStudentKIIM.load_from_checkpoint("/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/WA/Supervised/20/result_stats/checkpoints/epoch=13-val_iou_macro_irr=0.594.ckpt", strict=False, **cfg.model)
# #     # student_model = KIIM(...)
#     # model = TeacherStudentKIIM(teacher=teacher_model.student, student=teacher_model.student, num_classes=cfg.model.num_classes, alpha=cfg.train.alpha,alpha_decay=cfg.train.alpha_decay)
    
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
