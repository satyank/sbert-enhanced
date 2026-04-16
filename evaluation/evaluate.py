"""
evaluation/evaluate.py
----------------------
Evaluates a trained SBERT model on all 7 STS benchmarks.

The metric is Spearman correlation — it measures how well the ranking
of predicted similarity scores matches the ranking of gold scores.
A score of 1.0 is perfect, 0.0 is random.

Usage:
    python evaluation/evaluate.py --model_path experiments/pool-mean_best.pt
    python evaluation/evaluate.py --model_path experiments/pool-weighted_best.pt --pooling weighted
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

import torch
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm

from models.sbert import SentenceBERT


# Maps our benchmark names to their HuggingFace dataset IDs
BENCHMARK_MAP = {
    "STS12":        ("mteb/sts12-sts", "test"),
    "STS13":        ("mteb/sts13-sts", "test"),
    "STS14":        ("mteb/sts14-sts", "test"),
    "STS15":        ("mteb/sts15-sts", "test"),
    "STS16":        ("mteb/sts16-sts", "test"),
    "STSBenchmark": ("mteb/stsbenchmark-sts", "test"),
    "SICK-R":       ("mteb/sickr-sts", "test"),
}


def load_benchmark(benchmark_name: str, cache_dir: str = "data/cache") -> tuple:
    """
    Load one STS benchmark dataset.

    Returns:
        sentences1: list of strings
        sentences2: list of strings
        scores:     list of floats (gold similarity, scale varies by benchmark)
    """
    dataset_id, split = BENCHMARK_MAP[benchmark_name]
    ds = load_dataset(dataset_id, split=split, cache_dir=cache_dir)

    # All MTEB STS datasets use 'sentence1', 'sentence2', 'score'
    return ds["sentence1"], ds["sentence2"], ds["score"]


def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between matched pairs.

    For two embeddings u and v:
        cosine_similarity(u, v) = dot(u, v) / (||u|| * ||v||)

    Args:
        emb_a: (n, 768)
        emb_b: (n, 768)

    Returns:
        similarities: (n,) — one similarity score per pair
    """
    # Normalize each vector to unit length
    norm_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
    norm_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)

    # Dot product of matched pairs gives cosine similarity
    return (norm_a * norm_b).sum(axis=1)


def evaluate_benchmark(model: SentenceBERT, benchmark_name: str,
                        device: str = "cpu", cache_dir: str = "data/cache", max_samples=None) -> float:
    """
    Evaluate the model on one STS benchmark.

    Returns:
        spearman_corr: float between -1 and 1 (higher is better)
    """
    sentences1, sentences2, gold_scores = load_benchmark(benchmark_name, cache_dir)

    if max_samples is not None:
        sentences1 = sentences1[:max_samples]
        sentences2 = sentences2[:max_samples]
        gold_scores = gold_scores[:max_samples]
        print(f"  [LOCAL TEST] Using only {max_samples} eval examples")

    # Encode all sentences (in batches to avoid OOM)
    emb1 = model.encode_sentences(list(sentences1), batch_size=64, device=device).numpy()
    emb2 = model.encode_sentences(list(sentences2), batch_size=64, device=device).numpy()

    # Compute cosine similarity for each pair
    predicted_scores = cosine_similarity_matrix(emb1, emb2)

    # Spearman correlation compares the RANKING of scores, not absolute values
    # This is robust to different score scales across benchmarks
    correlation, _ = spearmanr(predicted_scores, gold_scores)

    return float(correlation)


def evaluate_all(model: SentenceBERT, device: str = "cpu",
                 cache_dir: str = "data/cache", max_samples=None) -> dict:
    """
    Run evaluation on all 7 benchmarks and print a formatted results table.

    Returns:
        results: dict mapping benchmark_name -> spearman_correlation
    """
    results = {}

    print(f"\n{'Benchmark':<20} {'Spearman':>12}")
    print("─" * 34)

    for benchmark_name in BENCHMARK_MAP:
        print(f"  Evaluating {benchmark_name}...", end="\r")
        score = evaluate_benchmark(model, benchmark_name, device, cache_dir, max_samples)
        results[benchmark_name] = score
        # Multiply by 100 to show as percentage (matches SBERT paper format)
        print(f"{benchmark_name:<20} {score * 100:>11.2f}")

    avg = np.mean(list(results.values())) * 100
    print("─" * 34)
    print(f"{'Average':<20} {avg:>11.2f}\n")

    return results


def analyze_token_weights(model: SentenceBERT, sentences: list,
                           device: str = "cpu") -> None:
    """
    Enhancement 2 analysis: print the attention weight assigned to each token.

    Use this to verify that content words (nouns, verbs, adjectives) get
    higher weights than stop words (the, a, is, of, ...).

    Only works with pooling_strategy='weighted'.
    """
    if model.pooling_strategy != "weighted":
        print("Token weight analysis only works with pooling_strategy='weighted'")
        return

    model.eval()
    print("\nToken weight analysis:")
    print("─" * 60)

    for sentence in sentences:
        tokens = model.tokenize([sentence])
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            bert_output = model.bert(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )
            weights = model.pooling.get_token_weights(
                bert_output.last_hidden_state,
                tokens["attention_mask"]
            )

        # Convert token IDs back to readable strings
        token_ids = tokens["input_ids"][0].tolist()
        token_strings = model.tokenizer.convert_ids_to_tokens(token_ids)
        weight_values = weights[0].cpu().numpy()

        print(f"\nSentence: '{sentence}'")
        print(f"{'Token':<20} {'Weight':>8}")
        print("─" * 30)

        for tok, w in zip(token_strings, weight_values):
            if tok in ("[PAD]",):  # skip padding
                continue
            # Show a bar to make weights visually intuitive
            bar = "█" * int(w * 200)
            print(f"{tok:<20} {w:>8.4f}  {bar}")

def compare_models(model_configs: list, device: str = "cpu",
                   cache_dir: str = "data/cache", max_samples: int = None) -> None:
    """
    Load multiple trained models and print a side-by-side comparison table.

    Args:
        model_configs: list of dicts, each with 'name', 'path', and 'pooling'
        device:        'cuda' or 'cpu'
        cache_dir:     where datasets are cached
        max_samples:   limit eval examples (for local testing)

    Example:
        compare_models([
            {"name": "Baseline",          "path": "experiments/baseline_best.pt",  "pooling": "mean"},
            {"name": "Multi-task",         "path": "experiments/multitask_best.pt", "pooling": "mean"},
            {"name": "Weighted pooling",   "path": "experiments/weighted_best.pt",  "pooling": "weighted"},
            {"name": "Both enhancements",  "path": "experiments/both_best.pt",      "pooling": "weighted"},
        ])
    """
    import yaml

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Collect results for every model
    all_results = {}

    for model_config in model_configs:
        name     = model_config["name"]
        path     = model_config["path"]
        pooling  = model_config["pooling"]

        print(f"\nEvaluating: {name}")
        print(f"  Loading from: {path}")

        # Load the model
        model = SentenceBERT(
            model_name=config["model"]["base_model"],
            pooling_strategy=pooling,
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        # Run evaluation on all benchmarks
        results = {}
        for benchmark_name in BENCHMARK_MAP:
            score = evaluate_benchmark(model, benchmark_name, device,
                                       cache_dir, max_samples)
            results[benchmark_name] = score
            print(f"  {benchmark_name:<20} {score * 100:.2f}")

        all_results[name] = results

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Print comparison table ────────────────────────────────────────────
    benchmarks = list(BENCHMARK_MAP.keys())
    model_names = list(all_results.keys())
    col_width = 16

    print("\n")
    print("=" * (22 + col_width * len(model_names)))
    print("RESULTS COMPARISON (Spearman x 100)")
    print("=" * (22 + col_width * len(model_names)))

    # Header row — model names
    header = f"{'Benchmark':<22}"
    for name in model_names:
        # Truncate long names so table stays aligned
        header += f"{name[:col_width-1]:<{col_width}}"
    print(header)
    print("-" * (22 + col_width * len(model_names)))

    # One row per benchmark
    for benchmark in benchmarks:
        row = f"{benchmark:<22}"
        scores = [all_results[name][benchmark] * 100 for name in model_names]
        best_score = max(scores)
        for score in scores:
            cell = f"{score:.2f}"
            # Mark the best score in each row with an asterisk
            if score == best_score:
                cell += "*"
            row += f"{cell:<{col_width}}"
        print(row)

    # Average row
    print("-" * (22 + col_width * len(model_names)))

    # Calculate ALL averages first before printing
    # This fixes the double asterisk bug where max(avgs)
    # was being checked against a growing list
    avgs = {}
    for name in model_names:
        avgs[name] = np.mean(list(all_results[name].values())) * 100

    # Find the single best average across all models
    best_avg = max(avgs.values())

    # Now print with correct single asterisk
    avg_row = f"{'Average':<22}"
    for name in model_names:
        avg = avgs[name]
        marker = "*" if avg == best_avg else ""
        avg_row += f"{avg:.2f}{marker:<{col_width - 5}}"
    print(avg_row)
    print("=" * (22 + col_width * len(model_names)))
    print("* = best score in that row")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SBERT model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a single model checkpoint for solo evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls", "weighted"])
    parser.add_argument("--analyze_weights", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all saved models side by side")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = config["data"]["cache_dir"]
    max_samples = config["evaluation"].get("max_eval_samples")
    save_dir = config["training"]["save_dir"]  # ← read save_dir from config

    if args.compare:
        save_dir = config["training"]["save_dir"]

        compare_models(
            model_configs=[
                {
                    "name": "Baseline (mean)",
                    "path": os.path.join(save_dir, "baseline_mean_best.pt"),
                    "pooling": "mean"
                },
                {
                    "name": "Max pooling",
                    "path": os.path.join(save_dir, "max_pooling_best.pt"),
                    "pooling": "max"
                },
                {
                    "name": "CLS pooling",
                    "path": os.path.join(save_dir, "cls_pooling_best.pt"),
                    "pooling": "cls"
                },
                {
                    "name": "Weighted pooling",
                    "path": os.path.join(save_dir, "weighted_pooling_best.pt"),
                    "pooling": "weighted"
                },
                {
                    "name": "Multitask (mean)",
                    "path": os.path.join(save_dir, "multitask_lam0.5_best.pt"),
                    "pooling": "mean"
                },
                {
                    "name": "Both enhancements",
                    "path": os.path.join(save_dir, "both_enhancements_best.pt"),
                    "pooling": "weighted"
                },
            ],
            device=device,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
    elif args.model_path:
        # Resolve model path — if just a filename, look in save_dir
        # If full path already given, use as is
        model_path = args.model_path if os.path.isabs(args.model_path) \
                     else os.path.join(save_dir, args.model_path)

        print(f"Loading model from: {model_path}")
        model = SentenceBERT(
            model_name=config["model"]["base_model"],
            pooling_strategy=args.pooling,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        results = evaluate_all(model, device=device, cache_dir=cache_dir,
                               max_samples=max_samples)

        if args.analyze_weights:
            example_sentences = [
                "The dog ran quickly through the park.",
                "A cat sat quietly on the windowsill.",
                "The financial markets experienced significant volatility.",
            ]
            analyze_token_weights(model, example_sentences, device)

    else:
        print("Please provide either --model_path for single evaluation or --compare for side by side comparison.")