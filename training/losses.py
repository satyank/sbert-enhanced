# this file implements the three loss functions used in SBERT paper
# NLIClassificationLoss: "Do these sentences entail, contradict, or are neutral?"
# STSRegressionLoss: "How similar are these sentences? (0.0 to 1.0)"
# TripletLoss: "Anchor should be closer to positive than negative"

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLIClassificationLoss(nn.Module):
    def __init__(self, hidden_size: int = 768, num_labels: int = 3):
        super().__init__()
        self.classifier = nn.Linear(3 * hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor):
        diff = torch.abs(emb_a - emb_b)
        combined = torch.cat([emb_a, emb_b, diff], dim=1)
        logits = self.classifier(combined)  # (batch_size, 3)
        return self.loss_fn(logits, labels)

# Computes cosine similarity between sentence embeddings and compares
# to the gold score (normalized from [0,5] to [0,1]).
class STSRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)  # (batch_size,)
        return self.loss_fn(cos_sim, scores)

# Triplet loss: trains the model so that an anchor sentence is
# closer to a positive (similar) sentence than a negative (dissimilar) one,
# by at least margin distance.
#
# Note: TripletLoss is implemented for completeness per the original paper
# but is not used in our training experiments
class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive, p=2)  # (batch_size,)
        d_neg = F.pairwise_distance(anchor, negative, p=2)  # (batch_size,)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean()
