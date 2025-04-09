import sys
sys.path.append('/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM')

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pathlib import Path
import torch
import pandas as pd
from data.data_module import IrrigationDataModule
from models.BaseModel import KIIM
from models.TeacherStudentModel_v1 import TeacherStudentKIIM

@hydra.main(config_path="/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/KIIM/config", config_name="Teacher-Training", version_base="1.2")
def evaluate(cfg: DictConfig) -> None:
    print(f"Running on GPUs: {cfg.train.devices}")
    pl.seed_everything(cfg.train.seed)

    # Load the data module (only for testing)
    data_module = IrrigationDataModule(cfg, merge_train_valid=False)

    # Initialize the student model
    student_model = KIIM(**cfg.model)

    # Load the trained model checkpoint
    checkpoint_path = "/project/biocomplexity/wyr6fx(Nibir)/IrrigationMapping/CO/UnSupervised/20/result_stats/checkpoints/epoch=11-val_iou_macro_irr=0.877.ckpt"
    # checkpoint = torch.load(checkpoint_path)
    # student_model.load_state_dict(checkpoint['state_dict'], strict=False)
    # print(f"Loaded model from checkpoint: {checkpoint_path}")
    
    # model = TeacherStudentKIIM(teacher=student_model, student=student_model, num_classes=cfg.model.num_classes)
    
    model = TeacherStudentKIIM.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    # Set up the trainer for testing
    strategy = DDPStrategy(find_unused_parameters=True) if len(cfg.train.devices) > 1 else 'ddp'
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=strategy,
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    try:
        # Run the test phase
        print("\nRunning test...\n")
        test_results = trainer.test(model, datamodule=data_module)
        test_df = pd.DataFrame(test_results)

        # Save the test results
        save_dir = Path(cfg.logging.run_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(save_dir / "test_results.csv", index=False)
        print(f"Test results saved at: {save_dir / 'test_results.csv'}")

    except Exception as e:
        print(f"Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate()
