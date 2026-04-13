"""
training/losses.py
------------------
The three loss functions used to train SBERT (as described in the original paper).
Owner: Ayman

Quick summary of what each loss does:
  NLIClassificationLoss : "Do these sentences entail, contradict, or are neutral?"
  STSRegressionLoss     : "How similar are these sentences? (0.0 to 1.0)"
  TripletLoss           : "Anchor should be closer to positive than negative"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NLIClassificationLoss(nn.Module):
    """
    Classification loss for Natural Language Inference training.

    Given embeddings u (sentence A) and v (sentence B), we build a combined
    feature vector [u, v, |u-v|] and classify it as one of three labels:
        0 = entailment   (A implies B)
        1 = neutral      (A and B are unrelated)
        2 = contradiction (A and B conflict)

    The |u-v| term captures the difference between the sentences,
    which is very informative for classification.
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 3):
        super().__init__()

        # Input size is 3x hidden because we concat [u, v, |u-v|]
        self.classifier = nn.Linear(3 * hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_a:  (batch_size, 768)
            emb_b:  (batch_size, 768)
            labels: (batch_size,) integers — 0, 1, or 2

        Returns:
            scalar cross-entropy loss
        """
        # Element-wise absolute difference captures how different the embeddings are
        diff = torch.abs(emb_a - emb_b)

        # Concatenate all three along the feature dimension -> (batch_size, 3*768)
        combined = torch.cat([emb_a, emb_b, diff], dim=1)

        # Linear layer maps to 3 class scores
        logits = self.classifier(combined)  # (batch_size, 3)

        return self.loss_fn(logits, labels)


class STSRegressionLoss(nn.Module):
    """
    Regression loss for Semantic Textual Similarity training.

    Computes cosine similarity between sentence embeddings and compares
    to the gold score (normalized from [0,5] to [0,1]).

    Cosine similarity is in [-1, 1] but for STS it's typically in [0, 1]
    since semantically related sentences tend to have positive similarity.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            emb_a:  (batch_size, 768)
            emb_b:  (batch_size, 768)
            scores: (batch_size,) gold scores already normalized to [0, 1]

        Returns:
            scalar MSE loss
        """
        # cosine_similarity returns values in [-1, 1]
        # dim=1 means we compare across the feature dimension (not batch)
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)  # (batch_size,)

        return self.loss_fn(cos_sim, scores)


class TripletLoss(nn.Module):
    """
    Triplet loss: trains the model so that an anchor sentence is
    closer to a positive (similar) sentence than a negative (dissimilar) one,
    by at least `margin` distance.

    Loss = mean( max(0,  d(anchor, positive) - d(anchor, negative) + margin) )

    If the model already separates positive and negative by more than margin,
    the loss is 0 (the term clamps to 0 via max).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor:   (batch_size, 768) — the reference sentence
            positive: (batch_size, 768) — a similar sentence to anchor
            negative: (batch_size, 768) — a dissimilar sentence to anchor

        Returns:
            scalar triplet loss
        """
        # Euclidean distance: sqrt(sum((a-b)^2)) for each pair in batch
        # p=2 means L2 (Euclidean) norm
        d_pos = F.pairwise_distance(anchor, positive, p=2)  # (batch_size,)
        d_neg = F.pairwise_distance(anchor, negative, p=2)  # (batch_size,)

        # Hinge loss: penalize when negative is not far enough from anchor
        # clamp(min=0) replaces negative values with 0
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)

        return loss.mean()
