# this script downloads and caches all datasets needed for training and evaluation.

import os
from datasets import load_dataset

# function to download the SNLI and MultiNLI datasets for NLI classification training
# Sample:
# - premise: first sentence
# - hypothesis: second sentence
# - label: can be one of the three integer values (0 = entailment, 1 = neutral, 2 = contradiction)
# we assign a value of -1 for inavlid labels
def download_nli_datasets(cache_dir: str = "data/cache"):
    os.makedirs(cache_dir, exist_ok=True)

    print("Downloading SNLI (~550k training examples)...")
    snli = load_dataset("snli", cache_dir=cache_dir)
    print(f"\tTrain: {len(snli['train'])} | Val: {len(snli['validation'])} | Test: {len(snli['test'])}")

    sample = snli["train"][0]
    print(f"\tSample — premise: '{sample['premise']}' | hypothesis: '{sample['hypothesis']}' | label: {sample['label']}")

    print("\nDownloading MultiNLI (~393k training examples)...")
    mnli = load_dataset("multi_nli", cache_dir=cache_dir)
    print(f"\tTrain: {len(mnli['train'])} | Val matched: {len(mnli['validation_matched'])}")

    sample = mnli["train"][0]
    print(f"\tSample — premise: '{sample['premise']}' | label: {sample['label']}")

# this function downloads the STS-B dataset
# sample contains:
# - sentence1, sentence2: the sentence pair
# - similarity_score:     float from 0.0 (unrelated) to 5.0 (identical)
# during training we normalize scores to [0, 1] by dividing by 5.
def download_sts_dataset(cache_dir: str = "data/cache"):
    os.makedirs(cache_dir, exist_ok=True)

    print("\nDownloading STS-B...")
    stsb = load_dataset("stsb_multi_mt", name="en", cache_dir=cache_dir)
    print(f"\tTrain: {len(stsb['train'])} | Val: {len(stsb['dev'])} | Test: {len(stsb['test'])}")

    sample = stsb["train"][0]
    print(f"\tSample — s1: '{sample['sentence1']}' | s2: '{sample['sentence2']}' | score: {sample['similarity_score']}")

# this function downloads the 7 STS evaluation datasets used in SBERT paper
# STS12, STS13, STS14, STS15, STS16, STSBenchmark, SICK-R
def download_eval_benchmarks(cache_dir: str = "data/cache"):
    os.makedirs(cache_dir, exist_ok=True)
    sts_datasets = {
        "STS12": ("mteb/sts12-sts", "test"),
        "STS13": ("mteb/sts13-sts", "test"),
        "STS14": ("mteb/sts14-sts", "test"),
        "STS15": ("mteb/sts15-sts", "test"),
        "STS16": ("mteb/sts16-sts", "test"),
        "STSBenchmark": ("mteb/stsbenchmark-sts", "test"),
        "SICK-R": ("mteb/sickr-sts", "test"),
    }

    print("\nDownloading evaluation benchmarks...")
    for name, (dataset_id, split) in sts_datasets.items():
        try:
            ds = load_dataset(dataset_id, split=split, cache_dir=cache_dir)
            print(f"\t{name}: {len(ds)} examples")
        except Exception as e:
            print(f"\t{name}: failed to download — {e}")


if __name__ == "__main__":
    download_nli_datasets()
    download_sts_dataset()
    download_eval_benchmarks()
    print("\nAll datasets ready.")
