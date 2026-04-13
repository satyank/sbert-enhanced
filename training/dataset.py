"""
training/dataset.py
-------------------
PyTorch Dataset classes for NLI and STS training data.

PyTorch DataLoaders need a Dataset object that implements:
    __len__()      -> how many examples total
    __getitem__(i) -> return example i as a dict
"""

import torch
from torch.utils.data import Dataset


class NLIDataset(Dataset):
    """
    Dataset for Natural Language Inference training.

    Each example: (premise, hypothesis, label)
    Label: 0=entailment, 1=neutral, 2=contradiction

    We combine SNLI and MultiNLI into one big dataset.
    Invalid labels (-1) are filtered out before this class is used.
    """

    def __init__(self, premises: list, hypotheses: list, labels: list):
        """
        Args:
            premises:    list of strings — the first sentence
            hypotheses:  list of strings — the second sentence
            labels:      list of ints   — 0, 1, or 2
        """
        assert len(premises) == len(hypotheses) == len(labels), \
            "All lists must have the same length"

        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

    def __len__(self) -> int:
        return len(self.premises)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sentence_a": self.premises[idx],
            "sentence_b": self.hypotheses[idx],
            "label": self.labels[idx],
        }


class STSDataset(Dataset):
    """
    Dataset for Semantic Textual Similarity regression training.

    Each example: (sentence1, sentence2, similarity_score)
    Scores are normalized from [0,5] to [0,1] before being stored here.
    """

    def __init__(self, sentences1: list, sentences2: list, scores: list):
        """
        Args:
            sentences1: list of strings
            sentences2: list of strings
            scores:     list of floats in [0, 1] (pre-normalized)
        """
        assert len(sentences1) == len(sentences2) == len(scores)

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores

    def __len__(self) -> int:
        return len(self.sentences1)

    def __getitem__(self, idx: int) -> dict:
        return {
            "sentence_a": self.sentences1[idx],
            "sentence_b": self.sentences2[idx],
            "score": float(self.scores[idx]),
        }


def collate_nli(batch: list, tokenizer, max_length: int = 128) -> dict:
    """
    Custom collate function for NLI batches.

    PyTorch's default collate can't tokenize strings, so we do it here.
    This function is called automatically by the DataLoader for each batch.

    Args:
        batch:      list of dicts from NLIDataset.__getitem__()
        tokenizer:  BERT tokenizer from the model
        max_length: truncate sequences longer than this

    Returns:
        dict with tokenized sentence_a, sentence_b, and labels tensor
    """
    sentences_a = [item["sentence_a"] for item in batch]
    sentences_b = [item["sentence_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    # Tokenize both sets of sentences
    encoded_a = tokenizer(sentences_a, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
    encoded_b = tokenizer(sentences_b, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")

    return {
        "sentence_a": encoded_a,
        "sentence_b": encoded_b,
        "labels": labels,
    }


def collate_sts(batch: list, tokenizer, max_length: int = 128) -> dict:
    """
    Custom collate function for STS batches.
    Same idea as collate_nli but for regression data.
    """
    sentences_a = [item["sentence_a"] for item in batch]
    sentences_b = [item["sentence_b"] for item in batch]
    scores = torch.tensor([item["score"] for item in batch], dtype=torch.float)

    encoded_a = tokenizer(sentences_a, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
    encoded_b = tokenizer(sentences_b, padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")

    return {
        "sentence_a": encoded_a,
        "sentence_b": encoded_b,
        "scores": scores,
    }
