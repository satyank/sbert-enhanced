"""
training/train.py
-----------------
Main training script. Handles both baseline and multi-task training.

Usage examples:
    # Baseline — NLI sequential training with mean pooling
    python training/train.py --config configs/config.yaml

    # Enhancement 1 — joint multi-task training
    python training/train.py --config configs/config.yaml --multitask --lambda_weight 0.5

    # Enhancement 2 — learned weighted pooling
    python training/train.py --config configs/config.yaml --pooling weighted

    # Both enhancements combined
    python training/train.py --config configs/config.yaml --multitask --pooling weighted
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from functools import partial

import yaml
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from models.sbert import SentenceBERT
from training.losses import NLIClassificationLoss, STSRegressionLoss
from training.dataset import NLIDataset, STSDataset, collate_nli, collate_sts


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Lock down all random number generators so experiments are reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Data loading ─────────────────────────────────────────────────────────────

def build_nli_dataloader(config: dict, tokenizer) -> DataLoader:
    """
    Build a DataLoader for NLI training.
    Combines SNLI (~550k) and MultiNLI (~393k) into one shuffled dataset.
    """
    cache_dir = config["data"]["cache_dir"]

    print("Loading SNLI...")
    snli = load_dataset("snli", split="train", cache_dir=cache_dir)
    # Filter out invalid labels (-1 appears in SNLI when annotators disagreed)
    snli = snli.filter(lambda x: x["label"] != -1)

    print("Loading MultiNLI...")
    mnli = load_dataset("multi_nli", split="train", cache_dir=cache_dir)
    mnli = mnli.filter(lambda x: x["label"] != -1)

    # Combine into one big dataset
    combined = concatenate_datasets([snli, mnli])
    print(f"Total NLI examples: {len(combined)}")

    max_samples = config["training"].get("max_train_samples")
    if max_samples is not None:
        combined = combined.select(range(max_samples))
        print(f"  [LOCAL TEST] Using only {max_samples} NLI examples")

    dataset = NLIDataset(
        premises=combined["premise"],
        hypotheses=combined["hypothesis"],
        labels=combined["label"],
    )

    # partial() lets us pass extra args (tokenizer, max_length) to the collate function
    collate_fn = partial(collate_nli, tokenizer=tokenizer,
                         max_length=config["training"]["max_seq_length"])

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,      # parallel data loading workers
        pin_memory=True,    # faster GPU transfers
    )


def build_sts_dataloader(config: dict, tokenizer) -> DataLoader:
    """
    Build a DataLoader for STS-B regression training.
    Scores are normalized from [0, 5] to [0, 1].
    """
    cache_dir = config["data"]["cache_dir"]

    print("Loading STS-B...")
    stsb = load_dataset("stsb_multi_mt", name="en", split="train", cache_dir=cache_dir)

    max_samples = config["training"].get("max_train_samples")
    if max_samples is not None:
        stsb = stsb.select(range(min(max_samples, len(stsb))))
        print(f"  [LOCAL TEST] Using only {max_samples} STS examples")

    dataset = STSDataset(
        sentences1=stsb["sentence1"],
        sentences2=stsb["sentence2"],
        # Normalize: divide by 5 to get [0, 1] range
        scores=[s / 5.0 for s in stsb["similarity_score"]],
    )

    collate_fn = partial(collate_sts, tokenizer=tokenizer,
                         max_length=config["training"]["max_seq_length"])

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )


# ── Training loops ───────────────────────────────────────────────────────────

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors inside a batch dict to the target device (GPU/CPU)."""
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            # sentence_a and sentence_b are nested dicts of tensors
            # explicitly check each value is a tensor before moving
            result[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def train_epoch_sequential(model, nli_loader, optimizer, scheduler,
                            nli_loss_fn, device) -> float:
    """
    Baseline training: one full pass over NLI data.
    This is the same approach as the original SBERT paper.
    """
    model.train()
    total_loss = 0.0

    progress = tqdm(nli_loader, desc="Training (NLI)", leave=False)

    for batch in progress:
        # Move entire batch to device
        batch = move_batch_to_device(batch, device)

        # Explicitly move sentence dicts to device
        sentence_a = {k: v.to(device) for k, v in batch["sentence_a"].items()}
        sentence_b = {k: v.to(device) for k, v in batch["sentence_b"].items()}

        # Forward pass using explicitly moved tensors
        emb_a, emb_b = model(sentence_a, sentence_b)

        # Compute NLI classification loss
        loss = nli_loss_fn(emb_a, emb_b, batch["labels"])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(nli_loader)


def train_epoch_multitask(model, nli_loader, sts_loader, optimizer, scheduler,
                           nli_loss_fn, sts_loss_fn, lambda_weight: float, device) -> float:
    """
    Enhancement 1: Joint multi-task training.
    Cycles through STS data repeatedly so NLI sets the pace.
    Combined loss = lambda_weight * nli_loss + (1 - lambda_weight) * sts_loss
    """
    model.train()
    total_loss = 0.0
    steps = 0

    # cycle() makes STS loop forever so it never runs out
    # NLI loader (14,720 batches) sets the total number of steps
    from itertools import cycle
    sts_cycled = cycle(sts_loader)

    num_steps = len(nli_loader)   # ← 14,720 not 90
    progress = tqdm(enumerate(nli_loader), desc="Training (Multi-task)",
                    total=num_steps, leave=False)

    for step, nli_batch in progress:
        sts_batch = next(sts_cycled)  # ← cycles through STS repeatedly

        # Move to device
        nli_batch = move_batch_to_device(nli_batch, device)
        sts_batch = move_batch_to_device(sts_batch, device)

        nli_sent_a = {k: v.to(device) for k, v in nli_batch["sentence_a"].items()}
        nli_sent_b = {k: v.to(device) for k, v in nli_batch["sentence_b"].items()}
        sts_sent_a = {k: v.to(device) for k, v in sts_batch["sentence_a"].items()}
        sts_sent_b = {k: v.to(device) for k, v in sts_batch["sentence_b"].items()}

        # NLI forward pass
        emb_a_nli, emb_b_nli = model(nli_sent_a, nli_sent_b)
        nli_loss = nli_loss_fn(emb_a_nli, emb_b_nli, nli_batch["labels"])

        # STS forward pass
        emb_a_sts, emb_b_sts = model(sts_sent_a, sts_sent_b)
        sts_loss = sts_loss_fn(emb_a_sts, emb_b_sts, sts_batch["scores"])

        # Combined loss with lambda weighting
        combined_loss = lambda_weight * nli_loss + (1 - lambda_weight) * sts_loss

        # Backward pass
        optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += combined_loss.item()
        steps += 1
        progress.set_postfix({
            "nli": f"{nli_loss.item():.4f}",
            "sts": f"{sts_loss.item():.4f}",
            "total": f"{combined_loss.item():.4f}",
        })

    return total_loss / max(steps, 1)


# ── Main training function ───────────────────────────────────────────────────

def train(config: dict, multitask: bool = False, lambda_weight: float = 0.5,
          pooling_strategy: str = "mean", run_name: str = "sbert",
          resume_from: str = None) -> SentenceBERT:
    """
    Full training pipeline. Returns the trained model.

    Two phases for baseline and weighted pooling (matching original SBERT paper):
      Phase 1: NLI classification training (4 epochs)
      Phase 2: STS regression fine-tuning  (1 epoch, lower LR)

    For multitask training:
      Single phase: NLI + STS mixed together every step (4 epochs)
    """
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Run: {run_name} | Pooling: {pooling_strategy} | Multi-task: {multitask}\n")

    # Initialize W&B logging
    wandb.init(
        project="sbert-reproduction",
        name=run_name,
        config={
            "pooling": pooling_strategy,
            "multitask": multitask,
            "lambda": lambda_weight,
            **config["training"],
        }
    )

    # Build model
    model = SentenceBERT(
        model_name=config["model"]["base_model"],
        pooling_strategy=pooling_strategy,
    ).to(device)

    # Build data loaders
    # Both NLI and STS are always loaded
    # NLI: main training data for all models
    # STS: used for multitask training AND phase 2 fine-tuning
    nli_loader = build_nli_dataloader(config, model.tokenizer)
    sts_loader = build_sts_dataloader(config, model.tokenizer)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        eps=1e-8,
    )

    # Total steps for scheduler
    total_steps = len(nli_loader) * config["training"]["epochs"]

    # Linear warmup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Loss functions
    nli_loss_fn = NLIClassificationLoss(hidden_size=model.hidden_size).to(device)
    sts_loss_fn = STSRegressionLoss().to(device)

    # Save directory
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Handle resume from checkpoint
    start_epoch = 1
    best_loss = float("inf")
    best_path = os.path.join(save_dir, f"{run_name}_best.pt")

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device))
        import re
        match = re.search(r'epoch(\d+)', resume_from)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"Resuming from epoch {start_epoch}")

    # ── Phase 1: NLI Training ──────────────────────────────────────────────
    print(f"\nPhase 1: NLI Training ({config['training']['epochs']} epochs)")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")

        if multitask:
            epoch_loss = train_epoch_multitask(
                model, nli_loader, sts_loader, optimizer, scheduler,
                nli_loss_fn, sts_loss_fn, lambda_weight, device
            )
        else:
            epoch_loss = train_epoch_sequential(
                model, nli_loader, optimizer, scheduler, nli_loss_fn, device
            )

        print(f"  Epoch {epoch} avg loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": epoch_loss})

        # Save checkpoint after every epoch
        ckpt_path = os.path.join(save_dir, f"{run_name}_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # Track best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_path)

    # ── Phase 2: STS Fine-tuning ───────────────────────────────────────────
    # Only for non-multitask models (baseline_mean and weighted_pooling)
    # Multitask already trains on STS simultaneously so skip it
    # This matches the original SBERT paper's two-phase training approach
    if not multitask:
        print("\nPhase 2: STS Fine-tuning (matching original SBERT paper)")

        # Lower learning rate for fine-tuning
        # Use 1/5 of original LR to make small gentle adjustments
        # to weights already trained in Phase 1
        finetune_lr = config["training"]["learning_rate"] / 5
        for param_group in optimizer.param_groups:
            param_group["lr"] = finetune_lr
        print(f"  Fine-tuning LR: {finetune_lr}")

        # Fresh scheduler for fine-tuning phase
        sts_total_steps = len(sts_loader)  # 1 epoch of STS
        sts_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=sts_total_steps,
        )

        # Fine-tune for 1 epoch on STS data
        model.train()
        total_sts_loss = 0.0
        steps = 0

        progress = tqdm(sts_loader, desc="STS Fine-tuning", leave=False)

        for batch in progress:
            batch = move_batch_to_device(batch, device)

            # Move sentence dicts explicitly to device
            sent_a = {k: v.to(device) for k, v in batch["sentence_a"].items()}
            sent_b = {k: v.to(device) for k, v in batch["sentence_b"].items()}

            # Forward pass
            emb_a, emb_b = model(sent_a, sent_b)

            # STS regression loss
            loss = sts_loss_fn(emb_a, emb_b, batch["scores"])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sts_scheduler.step()

            total_sts_loss += loss.item()
            steps += 1
            progress.set_postfix({"sts_loss": f"{loss.item():.4f}"})

        avg_sts_loss = total_sts_loss / max(steps, 1)
        print(f"  STS fine-tuning complete. Avg loss: {avg_sts_loss:.4f}")
        wandb.log({"sts_finetune_loss": avg_sts_loss})

        # Save final model after STS fine-tuning
        # This overwrites best_path with the fine-tuned version
        torch.save(model.state_dict(), best_path)
        print(f"  Fine-tuned model saved: {best_path}")

    wandb.finish()
    print(f"\nTraining complete. Best model saved to: {best_path}")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SBERT")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--multitask", action="store_true",
                        help="Enable Enhancement 1: joint multi-task training")
    parser.add_argument("--lambda_weight", type=float, default=0.5,
                        help="NLI loss weight in multi-task (0.0–1.0). Default: 0.5")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls", "weighted"],
                        help="Pooling strategy. 'weighted' enables Enhancement 2")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run (used in W&B and checkpoint filenames)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Auto-generate a descriptive run name if none given
    if args.run_name is None:
        name_parts = [f"pool-{args.pooling}"]
        if args.multitask:
            name_parts.append(f"multitask-lam{args.lambda_weight}")
        args.run_name = "_".join(name_parts)

    train(
        config=config,
        multitask=args.multitask,
        lambda_weight=args.lambda_weight,
        pooling_strategy=args.pooling,
        run_name=args.run_name,
    )
