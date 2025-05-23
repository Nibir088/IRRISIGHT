
# IRRISIGHT: A Multimodal Remote Sensing Dataset for Irrigation Mapping

**IRRISIGHT** is a large-scale, multimodal remote sensing dataset for irrigation classification, soil-water mapping, and agricultural monitoring. It spans over 20 U.S. states and provides more than 1.4 million ML-ready georeferenced patches with structured text prompts derived from soil, hydrology, land use, and climate data.

---

## 🌍 Overview

This repository supports training and evaluation of irrigation mapping models using:
- Sentinel-2 RGB and vegetation indices
- Crop/land/soil/geospatial metadata
- Vision-language prompts
- Supervised and semi-supervised pipelines

---

## 📦 Hugging Face Dataset

IRRISIGHT dataset is hosted at:  
🔗 https://huggingface.co/datasets/OBH30/IRRISIGHT

Each state directory (e.g., `Arizona/`) contains:
- `metadata.jsonl`: Structured metadata and text prompts
- `*.tar`: WebDataset shards with `.npy` patches and `.json` attributes

To load:
```python
from datasets import load_dataset
ds = load_dataset("OBH30/IRRISIGHT", split="train", streaming=True)
sample = next(iter(ds))
print(sample["image_path"], sample["text_prompt"])
```

---

## 📁 Repository Structure

```
IRRISIGHT/
├── Data/                          # WebDataset .tar files and metadata
│   ├── Arizona_0000.tar           # Sharded patches (RGB, indices, masks)
│   └── Arizona/metadata.jsonl     # Associated JSONL metadata
├── data/                          # WebDataset .tar files and metadata
│   ├── data_module.py             # custom data module for create batches
│   └── dataset_v2.py              # custom dataset
├── config/                        # Hydra configs for training and ablations
│   ├── supervised_training_gpu.yaml
│   └── dumps/                     # Teacher/Student configs
├── model_v3/                      # Modular model classes (CLIP, RemoteCLIP, BLIP, SAM, KIIM)
│   └── *.py
├── training_v2/                   # Scripts for training, label generation, evaluation
│   ├── Training_Teacher_Model-gpu.py
│   ├── Label_Generator_Unlabeled.py
│   └── Evaluate_Unlabeled.py
├── utils/, data/                  # Utilities and dataloaders
├── Evaluation.ipynb              # Evaluation notebook
├── validation.ipynb              # Additional analysis notebook
├── requirements.txt              # Python packages
├── environment.yml               # Conda environment file
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Nibir088/IRRISIGHT.git
cd IRRISIGHT
```

### 2. Create and activate environment
```bash
conda env create -f environment.yml
conda activate irrisight
```

### 3. Install Git LFS (for large files)
```bash
git lfs install
```

---

## 🚀 How to Run

### 🏋️ Train a supervised model
```bash
python training_v2/Training_Teacher_Model-gpu.py hydra.run.dir=outputs/ experiment=supervised_training_gpu
```

### ✨ Generate pseudo-labels for unlabeled data
```bash
python training_v2/Label_Generator_Unlabeled.py +experiment=generate_labels
```

### 📊 Evaluate model performance
```bash
python training_v2/Evaluate_Unlabeled.py +experiment=eval_unlabeled
```

---

## 📊 Sample Benchmark

| Model        | Modalities         | Flood | Sprinkler | Drip |
|--------------|--------------------|--------|------------|------|
| ResNet       | RGB                | 35.2   | 92.2       | 88.5 |
| SegFormer    | RGB                | 86.2   | 91.7       | 85.9 |
| CLIP         | RGB + Text Prompt  | 90.1   | 93.1       | 90.7 |
| RemoteCLIP   | RGB + Text Prompt  | 90.9   | 93.7       | 92.3 |
| **KIIM**     | RGB + Crop + Land  | **93.6** | **95.8** | **94.6** |

---

## 📸 Dataset Visuals

![Patch Example](assets/Sample_Patch_2.png)  
**Figure 1: Sample data patch with multimodal inputs.**

![Processing Pipeline](assets/Data_Processing.png)  
**Figure 2: End-to-end data processing pipeline.**

![Confidence Scores](assets/confidence_score.png)  
**Figure 3: Model confidence across states.**

![Dataset Coverage](assets/Evaluation.png)  
**Figure 4: Evaluation Framework.**

---

## 📝 License

Released for academic research use only. Contact authors for commercial use.

---

## 📬 Contact

Maintainer: [@Nibir088](https://github.com/Nibir088)  
Dataset: [OBH30/IRRISIGHT on Hugging Face](https://huggingface.co/datasets/OBH30/IRRISIGHT)

---
