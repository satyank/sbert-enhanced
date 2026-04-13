"""
models/pooling.py
-----------------
Pooling strategies that compress BERT's token-level outputs into one sentence vector.

Think of it like summarizing a paragraph into one sentence:
  - Mean pooling:     average every word equally
  - Max pooling:      take the most extreme value for each dimension
  - CLS pooling:      only use BERT's special summary token at position 0
  - Weighted pooling: (Enhancement 2) learn which words matter more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingLayer(nn.Module):

    def __init__(self, hidden_size: int = 768, pooling_strategy: str = "mean"):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.hidden_size = hidden_size

        # Enhancement 2: small trainable layer that scores each token.
        # Only created when strategy is 'weighted' — adds ~768 parameters total,
        # which is negligible compared to BERT's 110 million parameters.
        if pooling_strategy == "weighted":
            # Linear(768 -> 1): takes a token vector, outputs one importance score
            self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch_size, seq_len, 768) — one vector per token from BERT
            attention_mask:   (batch_size, seq_len)      — 1 = real token, 0 = padding

        Returns:
            sentence_embedding: (batch_size, 768) — one vector per sentence
        """
        if self.pooling_strategy == "mean":
            return self._mean_pool(token_embeddings, attention_mask)
        elif self.pooling_strategy == "max":
            return self._max_pool(token_embeddings, attention_mask)
        elif self.pooling_strategy == "cls":
            return self._cls_pool(token_embeddings)
        elif self.pooling_strategy == "weighted":
            return self._weighted_pool(token_embeddings, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Average all real token embeddings, ignoring padding.

        Example: sentence "Hello world [PAD] [PAD]"
          - mask = [1, 1, 0, 0]
          - we only average the first two tokens, not the padding
        """
        # Expand mask from (batch, seq_len) to (batch, seq_len, 768)
        # so we can multiply it with token_embeddings directly
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Zero out all padding token vectors
        masked_embeddings = token_embeddings * mask_expanded

        # Sum over tokens, then divide by count of real tokens
        sum_embeddings = masked_embeddings.sum(dim=1)
        token_count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # clamp avoids div-by-zero

        return sum_embeddings / token_count

    def _max_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Take the element-wise maximum across all real tokens.
        Padding positions are filled with -1e9 before taking max,
        so they can never win the max competition.
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Replace padding positions with a very large negative number
        token_embeddings_masked = token_embeddings.clone()
        token_embeddings_masked[mask_expanded == 0] = -1e9

        # Max across the sequence dimension
        return token_embeddings_masked.max(dim=1).values

    def _cls_pool(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Use only the [CLS] token — always at position 0.
        BERT is trained to pack sentence-level information into [CLS],
        but in practice mean pooling usually outperforms this.
        """
        return token_embeddings[:, 0, :]

    def _weighted_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Enhancement 2: Learned weighted pooling.

        Instead of treating "the" and "running" equally, this layer learns
        to assign higher weights to content words that carry more meaning.

        Steps:
          1. Score each token with a tiny linear layer -> shape (batch, seq_len, 1)
          2. Mask padding scores to -inf so they get ~0 weight after softmax
          3. Softmax over tokens -> weights that sum to 1 per sentence
          4. Weighted average of token vectors
        """
        # Step 1: score each token — shape becomes (batch_size, seq_len, 1)
        scores = self.attention_weights(token_embeddings)

        # Step 2: push padding positions to -inf so softmax gives them ~0 weight
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        scores = scores.masked_fill(mask == 0, -1e9)

        # Step 3: softmax across the seq_len dimension -> weights sum to 1
        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)

        # Step 4: weighted sum -> (batch_size, 768)
        return (token_embeddings * weights).sum(dim=1)

    def get_token_weights(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper for analysis: returns the raw attention weights for each token.
        Use this to inspect which words the model found important.

        Only works when pooling_strategy == 'weighted'.

        Returns:
            weights: (batch_size, seq_len) attention weight per token
        """
        if self.pooling_strategy != "weighted":
            raise ValueError("get_token_weights() only works with weighted pooling")

        scores = self.attention_weights(token_embeddings)
        mask = attention_mask.unsqueeze(-1).float()
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        return weights.squeeze(-1)           # (batch, seq_len)
