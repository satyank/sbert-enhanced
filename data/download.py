"""
data/download.py
----------------
Downloads and caches all datasets needed for training and evaluation.

Run once before training:
    python data/download.py
"""

import os
from datasets import load_dataset


def download_nli_datasets(cache_dir: str = "data/cache") -> None:
    """
    Download SNLI and MultiNLI — the two datasets used for NLI classification training.

    Each example contains:
        - premise:    the first sentence
        - hypothesis: the second sentence
        - label:      0 = entailment, 1 = neutral, 2 = contradiction, -1 = invalid (skip these)
    """
    os.makedirs(cache_dir, exist_ok=True)

    print("Downloading SNLI (~550k training examples)...")
    snli = load_dataset("snli", cache_dir=cache_dir)
    print(f"  Train: {len(snli['train'])} | Val: {len(snli['validation'])} | Test: {len(snli['test'])}")

    # Show a sample so you can verify the format looks right
    sample = snli["train"][0]
    print(f"  Sample — premise: '{sample['premise']}' | hypothesis: '{sample['hypothesis']}' | label: {sample['label']}")

    print("\nDownloading MultiNLI (~393k training examples)...")
    mnli = load_dataset("multi_nli", cache_dir=cache_dir)
    print(f"  Train: {len(mnli['train'])} | Val matched: {len(mnli['validation_matched'])}")

    sample = mnli["train"][0]
    print(f"  Sample — premise: '{sample['premise']}' | label: {sample['label']}")


def download_sts_dataset(cache_dir: str = "data/cache") -> None:
    """
    Download STS-B (Semantic Textual Similarity Benchmark).

    Each example contains:
        - sentence1, sentence2: the sentence pair
        - similarity_score:     float from 0.0 (unrelated) to 5.0 (identical)

    During training we normalize scores to [0, 1] by dividing by 5.
    """
    os.makedirs(cache_dir, exist_ok=True)

    print("\nDownloading STS-B...")
    stsb = load_dataset("stsb_multi_mt", name="en", cache_dir=cache_dir)
    print(f"  Train: {len(stsb['train'])} | Val: {len(stsb['dev'])} | Test: {len(stsb['test'])}")

    sample = stsb["train"][0]
    print(f"  Sample — s1: '{sample['sentence1']}' | s2: '{sample['sentence2']}' | score: {sample['similarity_score']}")


def download_eval_benchmarks(cache_dir: str = "data/cache") -> None:
    """
    Download the 7 STS evaluation benchmarks used in the original SBERT paper.
    We use the sentence-transformers benchmark data hosted on HuggingFace.

    Benchmarks: STS12, STS13, STS14, STS15, STS16, STSBenchmark, SICK-R
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Map our benchmark names to their HuggingFace dataset IDs
    benchmark_map = {
        "STS12":        ("mteb/sts12-sts", "test"),
        "STS13":        ("mteb/sts13-sts", "test"),
        "STS14":        ("mteb/sts14-sts", "test"),
        "STS15":        ("mteb/sts15-sts", "test"),
        "STS16":        ("mteb/sts16-sts", "test"),
        "STSBenchmark": ("mteb/stsbenchmark-sts", "test"),
        "SICK-R":       ("mteb/sickr-sts", "test"),
    }

    print("\nDownloading evaluation benchmarks...")
    for name, (dataset_id, split) in benchmark_map.items():
        try:
            ds = load_dataset(dataset_id, split=split, cache_dir=cache_dir)
            print(f"  {name}: {len(ds)} examples")
        except Exception as e:
            print(f"  {name}: failed to download — {e}")


if __name__ == "__main__":
    download_nli_datasets()
    download_sts_dataset()
    download_eval_benchmarks()
    print("\nAll datasets ready.")
