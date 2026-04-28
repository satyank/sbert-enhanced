# Reproducing Sentence-BERT with Two Training Enhancements

## Requirements

Python 3.8 or higher is required. Install all dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scipy>=1.11.0
numpy>=1.24.0
tqdm>=4.65.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
wandb>=0.16.0
```

---

## Google Colab Setup

This project was trained on Google Colab with an A100 GPU. Before running, mount Google Drive and clone the repository:

```python
from google.colab import drive
import os

drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/sbert-experiments', exist_ok=True)
```

Clone and navigate to the project:

```bash
git clone <your-repo-url>
cd sbert-enhanced
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Log in to Weights & Biases for experiment tracking:

```python
import wandb
wandb.login()
```

Verify GPU:

```python
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
```

---

## Configuration

All hyperparameters are in `configs/config.yaml`. Update `save_dir` to your Google Drive path before training:

```yaml
training:
  batch_size: 64
  learning_rate: 0.00002
  epochs: 4
  warmup_steps: 10000
  seed: 26
  max_seq_length: 128
  save_dir: "/content/drive/MyDrive/sbert-experiments/"
  max_train_samples: null   # set to a small number for local testing
  multitask:
    enabled: false
    lambda_weight: 0.5
```

For a quick local test set `max_train_samples: 10` and `batch_size: 2`.

---

## Data Download

Download all training and evaluation datasets:

```bash
python data/download.py
```

This downloads SNLI, MultiNLI, STS-B, and all seven evaluation benchmarks into `data/cache/`.

---

## Training

### Run all six models:

```bash
# Baseline — mean pooling, sequential training
python training/train.py --config configs/config.yaml --run_name baseline_mean

# Max pooling
python training/train.py --config configs/config.yaml --pooling max --run_name max_pooling

# CLS pooling
python training/train.py --config configs/config.yaml --pooling cls --run_name cls_pooling

# Enhancement 2 — learned weighted pooling
python training/train.py --config configs/config.yaml --pooling weighted --run_name weighted_pooling

# Enhancement 1 — joint multi-task training
python training/train.py --config configs/config.yaml --multitask --lambda_weight 0.5 --run_name multitask_lam0.5

# Both enhancements combined
python training/train.py --config configs/config.yaml --pooling weighted --multitask --lambda_weight 0.5 --run_name both_enhancements
```

### Arguments

| Argument | Description |
|---|---|
| `--config` | Path to config file (default: `configs/config.yaml`) |
| `--pooling` | Pooling strategy: `mean`, `max`, `cls`, `weighted` (default: `mean`) |
| `--multitask` | Enable joint multi-task training (Enhancement 1) |
| `--lambda_weight` | Balance between NLI and STS losses for multitask (default: `0.5`) |
| `--run_name` | Name for W&B run and saved model file |

---

## Evaluation

### Evaluate a single model:

```bash
python evaluation/evaluate.py --model_path baseline_mean_best.pt --pooling mean
python evaluation/evaluate.py --model_path multitask_lam0.5_best.pt --pooling mean
python evaluation/evaluate.py --model_path weighted_pooling_best.pt --pooling weighted
python evaluation/evaluate.py --model_path both_enhancements_best.pt --pooling weighted
python evaluation/evaluate.py --model_path max_pooling_best.pt --pooling max
python evaluation/evaluate.py --model_path cls_pooling_best.pt --pooling cls
```

### Compare all six models:

```bash
python evaluation/evaluate.py --compare
```

### Additional analysis:

```bash
# Token weight analysis (weighted pooling models only)
python evaluation/evaluate.py --model_path weighted_pooling_best.pt --pooling weighted --analyze_weights

# Error analysis
python evaluation/evaluate.py --error_analysis
```

---

## Project Structure

```
sbert-enhanced/
├── configs/
│   └── config.yaml          # All hyperparameters
├── data/
│   └── download.py          # Downloads all datasets
├── models/
│   ├── sbert.py             # SentenceBERT — siamese architecture
│   └── pooling.py           # Mean / Max / CLS / Weighted pooling
├── training/
│   ├── dataset.py           # NLIDataset, STSDataset, collate functions
│   ├── losses.py            # NLIClassificationLoss, STSRegressionLoss
│   └── train.py             # Training pipeline — sequential and multitask
├── evaluation/
│   └── evaluate.py          # Spearman evaluation, model comparison, error analysis
├── requirements.txt
└── README.md
```