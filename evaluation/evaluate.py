# this script will evaluate the trained SBERT models on the 7 STS dataset
# the evaluation metric is Spearman correlation. It measures how well the ranking
# of a predicted similarity scores matches te ranking of gold scores (0 being random and 1.0 being perfect)
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


# Maps the benchmark names to their HuggingFace dataset IDs
STS_DATASETS = {
    "STS12": ("mteb/sts12-sts", "test"),
    "STS13": ("mteb/sts13-sts", "test"),
    "STS14": ("mteb/sts14-sts", "test"),
    "STS15": ("mteb/sts15-sts", "test"),
    "STS16": ("mteb/sts16-sts", "test"),
    "STSBenchmark": ("mteb/stsbenchmark-sts", "test"),
    "SICK-R": ("mteb/sickr-sts", "test"),
}

# this method will load the specified dataset into the cache
def load_benchmark_dataset(dataset_name: str, cache_dir: str = "data/cache"):
    dataset_id, split = STS_DATASETS[dataset_name]
    ds = load_dataset(dataset_id, split = split, cache_dir = cache_dir)
    return ds["sentence1"], ds["sentence2"], ds["score"]

# this function will calculate the cosine similarity between two matching pairs
# cosine_similarity(u, v) = dot(u, v) / (||u|| * ||v||)
def cosine_similarity_matrix(embedding1: np.ndarray, embedding2: np.ndarray):
    # normalizing each vector to unit length
    norm_a = embedding1 / (np.linalg.norm(embedding1, axis=1, keepdims=True) + 1e-9)
    norm_b = embedding2 / (np.linalg.norm(embedding2, axis=1, keepdims=True) + 1e-9)
    return (norm_a * norm_b).sum(axis=1)


# this method will evaluate a given benchmark dataset and returns the spearman correlation value
# the spearman correlation value will be a float value between [-1, 1]
def evaluate_benchmark(model: SentenceBERT, dataset_name: str, device: str = "cpu", cache_dir: str = "data/cache", max_samples=None):
    sentences1, sentences2, gold_scores = load_benchmark_dataset(dataset_name, cache_dir)
    if max_samples is not None:
        sentences1 = sentences1[:max_samples]
        sentences2 = sentences2[:max_samples]
        gold_scores = gold_scores[:max_samples]
        print(f"\tUsing {max_samples} examples for evaluation")

    # encoding all sentences in batches to avoid OOM issue and converting to a ndarray
    emb1 = model.encode_sentences(list(sentences1), batch_size=64, device=device).numpy()
    emb2 = model.encode_sentences(list(sentences2), batch_size=64, device=device).numpy()

    predicted_cosine_similarity_scores = cosine_similarity_matrix(emb1, emb2)
    correlation, _ = spearmanr(predicted_cosine_similarity_scores, gold_scores)

    return float(correlation)

# will evaluate on all 7 benchmark datasets
def evaluate_all_benchmarks(model: SentenceBERT, device: str = "cpu", cache_dir: str = "data/cache", max_samples=None):
    results = {}

    print(f"\n{'Benchmark':<20} {'Spearman':>12}")
    print("─" * 34)
    for benchmark_name in STS_DATASETS:
        print(f"\tEvaluating {benchmark_name}...", end="\r")
        score = evaluate_benchmark(model, benchmark_name, device, cache_dir, max_samples)
        results[benchmark_name] = score
        print(f"{benchmark_name:<20} {score * 100:>11.2f}")

    avg = np.mean(list(results.values())) * 100
    print("─" * 34)
    print(f"{'Average':<20} {avg:>11.2f}\n")
    return results

# this function will analyze the token weights for weighted pooling strategy
def analyze_token_weights(model: SentenceBERT, sentences: list, device: str = "cpu"):
    if model.pooling_strategy != "weighted":
        print("Error... Cannot perform token weight analysis. Only works with pooling_strategy='weighted'")
        return
    
    model.eval()
    print("\nToken weights analysis:")
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

        token_ids = tokens["input_ids"][0].tolist()
        token_strings = model.tokenizer.convert_ids_to_tokens(token_ids)
        weight_values = weights[0].cpu().numpy()

        print(f"\nSentence: '{sentence}'")
        print(f"{'Token':<20} {'Weight':>8}")
        print("─" * 30)

        for tok, w in zip(token_strings, weight_values):
            if tok in ("[PAD]",):  # skipping padding
                continue
            print(f"{tok:<20} {w:>8.4f}")


# compare all the trained models in a side-by-side manner
def compare_models(model_configs: list, device: str = "cpu", cache_dir: str = "data/cache", max_samples: int = None):
    import yaml
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    all_results = {}
    for model_config in model_configs:
        name = model_config["name"]
        path = model_config["path"]
        pooling = model_config["pooling"]

        print(f"\nEvaluating: {name}")
        print(f"\tLoading from: {path}")
        model = SentenceBERT(
            model_name = config["model"]["base_model"],
            pooling_strategy = pooling,
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        results = {}
        for benchmark_name in STS_DATASETS:
            score = evaluate_benchmark(model, benchmark_name, device, cache_dir, max_samples)
            results[benchmark_name] = score
            print(f"\t{benchmark_name:<20} {score * 100:.2f}")
        all_results[name] = results

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # printing comparison table
    benchmarks = list(STS_DATASETS.keys())
    model_names = list(all_results.keys())
    col_width = 20

    print("\n")
    print("=" * (22 + col_width * len(model_names)))
    print("RESULTS COMPARISON (Spearman x 100)")
    print("=" * (22 + col_width * len(model_names)))

    header = f"{'Benchmark':<22}"
    for name in model_names:
        header += f"{name:<{col_width}}"
    print(header)
    print("-" * (22 + col_width * len(model_names)))

    for benchmark in benchmarks:
        row = f"{benchmark:<22}"
        scores = [all_results[name][benchmark] * 100 for name in model_names]
        best_score = max(scores)
        for score in scores:
            cell = f"{score:.2f}"
            if score == best_score:
                cell += "*"  # marking the best score in each row with an asterisk
            row += f"{cell:<{col_width}}"
        print(row)

    # average row
    print("-" * (22 + col_width * len(model_names)))
    avgs = {}
    for name in model_names:
        avgs[name] = np.mean(list(all_results[name].values())) * 100
    best_avg = max(avgs.values())

    avg_row = f"{'Average':<22}"
    for name in model_names:
        avg = avgs[name]
        marker = "*" if avg == best_avg else ""
        avg_row += f"{avg:.2f}{marker:<{col_width - 5}}"
    print(avg_row)
    print("=" * (22 + col_width * len(model_names)))
    print("* = best performing model")


# this is to find sentnece pairs multitask and baseline models disagree the most
def error_analysis(baseline_path: str, multitask_path: str, benchmark_name: str, config: dict, device: str = "cpu", n_examples: int = 10):
    cache_dir = config["data"]["cache_dir"]

    print(f"\nError Analysis — {benchmark_name}")
    print(f"Comparing baseline vs multitask")
    print("=" * 80)

    print("Loading baseline model...")
    model_baseline = SentenceBERT(
        model_name=config["model"]["base_model"],
        pooling_strategy="mean"
    )
    model_baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    model_baseline.to(device)
    model_baseline.eval()

    print("Loading multitask model...")
    model_multitask = SentenceBERT(
        model_name=config["model"]["base_model"],
        pooling_strategy="mean"
    )
    model_multitask.load_state_dict(torch.load(multitask_path, map_location=device))
    model_multitask.to(device)
    model_multitask.eval()

    sentences1, sentences2, gold_scores = load_benchmark_dataset(benchmark_name, cache_dir)

    print("Computing embeddings...")
    emb1_base = model_baseline.encode_sentences(list(sentences1), batch_size=64, device=device).numpy()
    emb2_base = model_baseline.encode_sentences(list(sentences2), batch_size=64, device=device).numpy()

    emb1_multi = model_multitask.encode_sentences(list(sentences1), batch_size=64, device=device).numpy()
    emb2_multi = model_multitask.encode_sentences(list(sentences2), batch_size=64, device=device).numpy()

    pred_base = cosine_similarity_matrix(emb1_base, emb2_base)
    pred_enh = cosine_similarity_matrix(emb1_multi, emb2_multi)

    gold = np.array(gold_scores)
    gold_norm = (gold - gold.min()) / (gold.max() - gold.min() + 1e-9)

    # finding pairs where models disagree most
    disagreement = np.abs(pred_base - pred_enh)
    top_indices = np.argsort(disagreement)[-n_examples:][::-1]

    print(f"\nTop {n_examples} sentence pairs where models disagree:")
    print("=" * 80)
    multitask_wins = 0
    baseline_wins = 0

    for rank, idx in enumerate(top_indices, 1):
        idx = int(idx)
        base_error = abs(pred_base[idx] - gold_norm[idx])
        multi_error  = abs(pred_enh[idx]  - gold_norm[idx])
        winner = "Multitask" if multi_error < base_error else "Baseline"

        if multi_error < base_error:
            multitask_wins += 1
        else:
            baseline_wins += 1

        print(f"\nExample {rank}:")
        print(f"\tSentence 1:        {sentences1[idx]}")
        print(f"\tSentence 2:        {sentences2[idx]}")
        print(f"\tGold score:        {gold_scores[idx]:.2f}")
        print(f"\tBaseline pred:     {pred_base[idx]:.4f}")
        print(f"\tMultitask pred:    {pred_enh[idx]:.4f}")
        print(f"\tDisagreement:      {disagreement[idx]:.4f}")
        print(f"\tCloser to gold:    {winner}")
        print("-" * 80)

    print(f"\nSummary:")
    print(f"\tMultitask closer to gold:  {multitask_wins}/{n_examples}")
    print(f"\tBaseline closer to gold:   {baseline_wins}/{n_examples}")
    print(f"\tMultitask win rate:        {multitask_wins/n_examples*100:.0f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SBERT model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a single model checkpoint for solo evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls", "weighted"])
    parser.add_argument("--analyze_weights", action="store_true",
                        help="Print token attention weights (only for weighted pooling)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all saved models side by side")
    parser.add_argument("--error_analysis", action="store_true",
                        help="Run error analysis between baseline and multitask")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = config["data"]["cache_dir"]
    max_samples = config["evaluation"].get("max_eval_samples")
    save_dir = config["training"]["save_dir"]

    if args.compare:
        compare_models(
            model_configs=[
                {
                    "name": "Baseline (Mean)",
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
                    "name": "Weighted pooling (E1)",
                    "path": os.path.join(save_dir, "weighted_pooling_best.pt"),
                    "pooling": "weighted"
                },
                {
                    "name": "Multitask (E2)",
                    "path": os.path.join(save_dir, "multitask_lam0.5_best.pt"),
                    "pooling": "mean"
                },
                {
                    "name": "Both enhancements (E1 + E2)",
                    "path": os.path.join(save_dir, "both_enhancements_best.pt"),
                    "pooling": "weighted"
                },
            ],
            device=device,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
    elif args.error_analysis:
        error_analysis(
            baseline_path=os.path.join(save_dir, "baseline_mean_best.pt"),
            multitask_path=os.path.join(save_dir, "multitask_lam0.5_best.pt"),
            benchmark_name="STSBenchmark",
            config=config,
            device=device,
            n_examples=10,
        )
    elif args.model_path:
        if os.path.isabs(args.model_path) or os.path.exists(args.model_path):
            model_path = args.model_path
        else:
            model_path = os.path.join(save_dir, args.model_path)

        if not os.path.exists(model_path):
            print(f"Error: checkpoint not found at {model_path}")
            print(f"Files in {save_dir}:")
            for f in sorted(os.listdir(save_dir)):
                print(f"  {f}")
            exit(1)

        print(f"Loading model from: {model_path}")
        model = SentenceBERT(model_name=config["model"]["base_model"], pooling_strategy=args.pooling,)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        results = evaluate_all_benchmarks(model, device=device, cache_dir=cache_dir, max_samples=max_samples)

        if args.analyze_weights:
            example_sentences = [
                "The dog ran quickly through the park.",
                "A cat sat quietly on the windowsill.",
                "The financial markets experienced significant volatility.",
                "Scientists discovered a new species of bird in the Amazon.",
                "The children played happily in the garden."
            ]
            analyze_token_weights(model, example_sentences, device)
    else:
        print("Please provide one of:")
        print("\t--model_path  for single model evaluation")
        print("\t--compare     for side by side comparison")
        print("\t--error_analysis  for baseline vs multitask analysis")
        print("\nAvailable checkpoints:")
        if os.path.exists(save_dir):
            for f in sorted(os.listdir(save_dir)):
                if f.endswith("_best.pt"):
                    print(f"\t{f}")