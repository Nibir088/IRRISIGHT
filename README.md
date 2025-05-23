# IRRISIGHT

## Project Structure
```
lightning_project/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Custom dataset definitions
│   ├── data_module.py      # LightningDataModule for data management
├── models/
│   ├── __init__.py
│   ├── lit_model.py        # PyTorch Lightning model definition
├── training/
│   ├── __init__.py
│   ├── train.py           # Script to train the model
│   ├── test.py            # Script to test the model
├── utils/
│   ├── __init__.py
│   ├── callbacks.py       # Custom callbacks
│   ├── metrics.py         # Additional metric implementations
├── scripts/
│   ├── run_training.sh    # Bash script to run training
├── config/
│   ├── config.yaml        # Configuration file
├── README.md
└── requirements.txt
```

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Modify the configuration in `config/supervised_training_gpu.yaml`
2. Run training:
   ```bash
   bash training_v2/Training_Teacher_Model-gpu.py
   ```

## License
[Your License]
