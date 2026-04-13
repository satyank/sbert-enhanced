# Sentence-BERT: Reproduction & Enhancements

**CS NLP Project** — The University of Texas at Dallas  
Ayman Shehzad Awal · Ege Berk Konya · Satyank Nadimpalli

---

## Overview

Reproduces [Sentence-BERT (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084) and adds two enhancements:

1. **Joint Multi-Task Training** (Enhancement 1) — train NLI and STS simultaneously
2. **Learned Weighted Pooling** (Enhancement 2) — replace mean pooling with trainable attention

---

## Project Structure

```
sbert-project/
├── configs/config.yaml          ← all hyperparameters in one place
├── data/download.py             ← download all datasets (run once)
├── models/
│   ├── sbert.py                 ← siamese BERT model
│   └── pooling.py               ← mean / max / cls / weighted pooling
├── training/
│   ├── dataset.py               ← PyTorch Dataset classes
│   ├── losses.py                ← NLI, STS regression, triplet losses
│   └── train.py                 ← main training script
├── evaluation/evaluate.py       ← Spearman eval on 7 benchmarks
├── experiments/                 ← checkpoints saved here (gitignored)
├── run_experiments.sh           ← runs all 4 variants end-to-end
└── requirements.txt
```

---

## Azure Setup (Step-by-Step)

### Step 1 — Activate your Azure student account
1. Go to [azure.microsoft.com/en-us/free/students](https://azure.microsoft.com/en-us/free/students)
2. Sign in with your **UTD email** (@utdallas.edu)
3. Follow the prompts — you'll get $100 free credits, no credit card needed

### Step 2 — Create a GPU Virtual Machine
1. In the Azure portal, click **Create a resource → Virtual Machine**
2. Choose these settings:
   - **Region**: East US (usually has T4 availability)
   - **Image**: Ubuntu 22.04 LTS
   - **Size**: `Standard_NC4as_T4_v3` (1× T4 GPU, ~$0.53/hr)
   - **Authentication**: SSH public key (generate one or use an existing key)
3. Click **Review + Create → Create**
4. Once deployed, copy the **Public IP address** from the VM overview page

### Step 3 — Connect to your VM
```bash
# From your laptop terminal (Mac/Linux)
ssh azureuser@YOUR_VM_PUBLIC_IP

# On Windows, use PowerShell or install Windows Terminal
```

### Step 4 — Set up the environment on the VM
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone your GitHub repo
git clone https://github.com/YOUR_USERNAME/sbert-project.git
cd sbert-project

# Install Python packages
pip3 install -r requirements.txt

# Verify GPU is detected — should show your T4
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Step 5 — Download datasets
```bash
# This downloads ~5GB of data — do it once and it caches automatically
python3 data/download.py
```

### Step 6 — Log into Weights & Biases
```bash
wandb login
# Paste your API key from wandb.ai/authorize
# Now all training runs will appear in your W&B dashboard automatically
```

### Step 7 — Run training
```bash
# Option A: Run all experiments automatically (recommended — leave overnight)
chmod +x run_experiments.sh
nohup ./run_experiments.sh > training.log 2>&1 &
# 'nohup' keeps it running even if your SSH connection drops
# Check progress: tail -f training.log

# Option B: Run one experiment manually
python3 training/train.py --config configs/config.yaml --pooling mean
python3 training/train.py --config configs/config.yaml --pooling weighted
python3 training/train.py --config configs/config.yaml --multitask --lambda_weight 0.5
```

### Step 8 — Evaluate
```bash
python3 evaluation/evaluate.py \
    --model_path experiments/baseline_mean_best.pt \
    --pooling mean

# For weighted pooling — also shows token weights (great for the demo!)
python3 evaluation/evaluate.py \
    --model_path experiments/weighted_pooling_best.pt \
    --pooling weighted \
    --analyze_weights
```

### Step 9 — IMPORTANT: Stop your VM when done
```bash
# In the Azure portal: go to your VM → click Stop
# Or from the terminal:
az vm deallocate --resource-group YOUR_RG --name YOUR_VM_NAME
```
> ⚠️ A running VM costs money even if you're not using it. Always stop it!

---

## Running All 4 Experiments

| Command | Description |
|---|---|
| `python training/train.py --pooling mean` | Baseline (mean pooling) |
| `python training/train.py --pooling mean --multitask --lambda_weight 0.5` | Enhancement 1 |
| `python training/train.py --pooling weighted` | Enhancement 2 |
| `python training/train.py --pooling weighted --multitask --lambda_weight 0.5` | Both |

---

## Expected Results Table

| Model | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg |
|---|---|---|---|---|---|---|---|---|
| SBERT (paper) | 70.97 | 76.53 | 73.19 | 79.09 | 74.30 | 76.55 | 72.05 | 74.67 |
| Ours: mean pooling | | | | | | | | |
| Ours: + multi-task | | | | | | | | |
| Ours: + weighted pooling | | | | | | | | |
| Ours: both | | | | | | | | |

---

## References

- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
- Devlin et al. (2019). *BERT.* NAACL.
- Yang et al. (2016). *Hierarchical Attention Networks.* NAACL.
