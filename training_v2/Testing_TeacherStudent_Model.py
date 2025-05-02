import sys
sys.path.append('/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM')

import torch
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.strategies import DDPStrategy
from data.data_module import IrrigationDataModule
from models_v2.TeacherStudentModel import TeacherStudentModel
from utils.train_config import save_experiment_config

@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM/config", config_name="Teacher-Student-Training_v1", version_base="1.2")
def test(cfg: DictConfig) -> None:
    print(f"Running test on GPUs: {cfg.train.devices}")
    pl.seed_everything(cfg.train.seed)

    # Load datamodule
    data_module = IrrigationDataModule(cfg, merge_train_valid=False)
    data_module.setup('test')

    # Load model
    model = TeacherStudentModel.load_from_checkpoint(cfg.eval.checkpoint_path, **cfg)
    print("Loaded model from:", cfg.eval.checkpoint_path)

    if len(cfg.train.devices) > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Setup trainer
    strategy = DDPStrategy(find_unused_parameters=False) if len(cfg.train.devices) > 1 else 'ddp'
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=strategy,
        precision=cfg.train.precision,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    # Run test
    test_results = trainer.test(model, datamodule=data_module)
    save_dir = Path(cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(save_dir / "test_results.csv", index=False)
    print("Saved test results to:", save_dir / "test_results.csv")

if __name__ == "__main__":
    test()
