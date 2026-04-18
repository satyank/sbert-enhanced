# this file imlements Dataset objects needed for PyTorch DataLoaders

import torch
from torch.utils.data import Dataset


class NLIDataset(Dataset):
    def __init__(self, premises: list, hypotheses: list, labels: list):
        assert len(premises) == len(hypotheses) == len(labels), "All lists must have the same length"
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx: int):
        return {
            "sentence_a": self.premises[idx],
            "sentence_b": self.hypotheses[idx],
            "label": self.labels[idx]
        }


class STSDataset(Dataset):
    def __init__(self, sentences1: list, sentences2: list, scores: list):
        assert len(sentences1) == len(sentences2) == len(scores)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, idx: int):
        return {
            "sentence_a": self.sentences1[idx],
            "sentence_b": self.sentences2[idx],
            "score": float(self.scores[idx]),
        }


# PyTorch's default collate can't tokenize strings, so we do it here.
# this function is called automatically by the DataLoader for each batch.
def collate_nli(batch: list, tokenizer, max_length: int = 128):
    sentences_a = [item["sentence_a"] for item in batch]
    sentences_b = [item["sentence_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encoded_a = tokenizer(sentences_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded_b = tokenizer(sentences_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return {
        "sentence_a": encoded_a,
        "sentence_b": encoded_b,
        "labels": labels,
    }

# same method as collate_nli but for regression data.
def collate_sts(batch: list, tokenizer, max_length: int = 128):
    sentences_a = [item["sentence_a"] for item in batch]
    sentences_b = [item["sentence_b"] for item in batch]
    scores = torch.tensor([item["score"] for item in batch], dtype=torch.float)
    encoded_a = tokenizer(sentences_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded_b = tokenizer(sentences_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return {
        "sentence_a": encoded_a,
        "sentence_b": encoded_b,
        "scores": scores,
    }
