# the purpose of this class is to compress BERT's token-level outputs into one sentence vector
# Mean pooling: average every word equally
# Max pooling: take the most extreme value for each dimension
# CLS pooling: only use BERT's special summary token at position 0
# Weighted pooling: (Enhancement 2) learn which words matter more

import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolingLayer(nn.Module):
    def __init__(self, hidden_size: int = 768, pooling_strategy: str = "mean"):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        self.hidden_size = hidden_size

        # Enhancement 2: small trainable layer that scores each token.
        # only created when strategy is 'weighted' and adds 768 parameters total,
        # which is negligible compared to BERT's 110 million parameters.
        if pooling_strategy == "weighted":
            self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
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

    # mean pooling - takes average of all token embeddings
    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        token_count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # clamp avoids div-by-zero
        return sum_embeddings / token_count

    # max pooling - takes element-wise maximum across all tokens
    # paddings are filled with -1e9 before taking max
    def _max_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings_masked = token_embeddings.clone()
        token_embeddings_masked[mask_expanded == 0] = -1e9
        return token_embeddings_masked.max(dim=1).values

    def _cls_pool(self, token_embeddings: torch.Tensor):
        return token_embeddings[:, 0, :]

    # enhancement 2: learned weighted pooling
    # this layer learns to assign higher weights to words which carry more meaning
    # 1. Score each token with a tiny linear layer -> shape (batch, seq_len, 1)
    # 2. Mask padding scores to -inf so they get ~0 weight after softmax
    # 3. Softmax over tokens -> weights that sum to 1 per sentence
    # 4. Weighted average of token vectors
    def _weighted_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention_weights(token_embeddings)
        mask = attention_mask.unsqueeze(-1).float() # (batch, seq_len, 1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1) # (batch_size, seq_len, 1)
        return (token_embeddings * weights).sum(dim=1)

    def get_token_weights(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy != "weighted":
            raise ValueError("get_token_weights() only works with weighted pooling")

        scores = self.attention_weights(token_embeddings)
        mask = attention_mask.unsqueeze(-1).float()
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1) # (batch, seq_len, 1)
        return weights.squeeze(-1) # (batch, seq_len)
