import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

from model_v3.TeacherModel import TeacherModel
from data.data_module_v2 import IrrigationDataModule
from omegaconf import OmegaConf

# === CONFIG ===
state = 'New Jersey'
cfg_path = '/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_Irrigation_Mapping_Model/Output/cross-state/vision/kiim/result_stats/configs/hydra_config.yaml'
ckpt_path = '/sfs/gpfs/tardis/project/bii_nssac/people/wyr6fx/NeurIPS_Irrigation_Mapping_Model/Output/cross-state/vision/kiim/result_stats/checkpoints/epoch=17-val_iou_macro_irr=0.912.ckpt'
save_dir = Path(f"/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_irrigation_data/generated_label/{state}")
save_dir.mkdir(parents=True, exist_ok=True)

# === SETUP ===
print(f"Generating pseudo-labels for {state}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = OmegaConf.load(cfg_path)
cfg.dataset.train_type = 'unsupervised'
cfg.dataset.states = [[state, 1]]

data_module = IrrigationDataModule(cfg)
data_module.setup('fit')
data_module.setup('test')

model = TeacherModel.load_from_checkpoint(ckpt_path, **cfg)
model.to(device)
model.eval()

# === INFERENCE LOOP ===
for batch_idx, batch in enumerate(tqdm(data_module.train_dataloader(), desc="Generating pseudo-labels")):
    with torch.no_grad():
        image_paths = batch['image_path']
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        probs = model(batch)['predictions']  # shape: [B, C, H, W]
        # probs = F.softmax(logits, dim=1)      # [B, C, H, W]
        conf, preds = torch.max(probs, dim=1) # [B, H, W] - class & confidence

    for i in range(preds.shape[0]):
        # patch_id = f"{state}_{batch_idx:05}_{i:02}"
        # print(batch.keys())
        image_path = image_paths[i]
        label_path = image_path.replace('patch', 'generated_label').replace('.tif', '.npy')
        conf_path = image_path.replace('patch', 'generated_conf').replace('.tif', '.npy')
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        os.makedirs(os.path.dirname(conf_path), exist_ok=True)
        

        label = preds[i].cpu().numpy().astype(np.uint8)
        conf_map = conf[i].cpu().numpy().astype(np.float32)

        # === Save .npy format ===
        np.save(label_path, label)
        np.save(conf_path, conf_map)

        # === Save optional PNG for visualization ===
        # Image.fromarray(label).save(save_dir / f"{patch_id}_label.png")
        # Image.fromarray((conf_map * 255).astype(np.uint8)).save(save_dir / f"{patch_id}_conf.png")

print("✅ Pseudo-label generation complete.")
