import sys
sys.path.append('/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM')

import torch
import hydra
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
from data.data_module import IrrigationDataModule
from models_v2.TeacherModel import TeacherModel
from utils.train_config import save_experiment_config

@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM/config", config_name="Teacher-Student-Training_v1", version_base="1.2")
def test(cfg: DictConfig) -> None:
    print(f"Running test on GPUs: {cfg.train.devices}")
    pl.seed_everything(cfg.train.seed)

    # Set up data
    data_module = IrrigationDataModule(cfg, merge_train_valid=False)
    data_module.setup('fit')
    data_module.setup('test')
    print(data_module.train_dataset.image_paths[:data_module.train_len])

#     # Load model checkpoint
#     model = TeacherModel.load_from_checkpoint(cfg.eval.checkpoint_path, **cfg)
#     print("Loaded model from:", cfg.eval.checkpoint_path)

#     if len(cfg.train.devices) > 1:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

#     # Configure trainer
#     strategy = DDPStrategy(find_unused_parameters=False) if len(cfg.train.devices) > 1 else 'ddp'
#     trainer = pl.Trainer(
#         accelerator=cfg.train.accelerator,
#         devices=cfg.train.devices,
#         strategy=strategy,
#         precision=cfg.train.precision,
#         log_every_n_steps=5,
#         enable_progress_bar=True,
#     )

#     # Run test
#     test_results = trainer.test(model, datamodule=data_module)
#     save_dir = Path(cfg.logging.run_name)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     test_df = pd.DataFrame(test_results)
#     test_df.to_csv(save_dir / "test_results.csv", index=False)
#     print("Test results saved to:", save_dir / "test_results.csv")

#     # Optionally save other metrics
#     save_experiment_config(cfg, data_module, model, trainer, save_dir)
#     if hasattr(model, 'get_metrics'):
#         final_metrics = model.get_metrics()
#         from omegaconf import OmegaConf
#         OmegaConf.save(config=final_metrics, f=save_dir / "final_metrics.yaml")

if __name__ == "__main__":
    test()
