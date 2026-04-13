"""
models/sbert.py
---------------
The core Sentence-BERT model.
Owner: Ayman

Architecture (siamese = "twin"):
    Sentence A ──► BERT ──► Pooling ──► vector_a ──►
                                                      compare (cosine similarity)
    Sentence B ──► BERT ──► Pooling ──► vector_b ──►

Both sentences share the EXACT same BERT weights — that's what "siamese" means.
This means we only need one BERT model in memory, not two.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from models.pooling import PoolingLayer


class SentenceBERT(nn.Module):

    def __init__(self, model_name: str = "bert-base-uncased", pooling_strategy: str = "mean"):
        super().__init__()

        # Load pre-trained BERT — this downloads ~440MB the first time
        print(f"Loading BERT model: {model_name}")
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # BERT-base outputs 768-dimensional vectors for each token
        self.hidden_size = self.bert.config.hidden_size  # 768

        # Plug in whichever pooling strategy was requested
        self.pooling = PoolingLayer(
            hidden_size=self.hidden_size,
            pooling_strategy=pooling_strategy,
        )
        self.pooling_strategy = pooling_strategy
        print(f"Pooling strategy: {pooling_strategy}")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenized input into a single sentence embedding.

        Args:
            input_ids:      (batch_size, seq_len) integer token IDs
            attention_mask: (batch_size, seq_len) 1=real token, 0=padding

        Returns:
            embeddings: (batch_size, 768)
        """
        # BERT forward pass — returns a named tuple with last_hidden_state and pooler_output
        # last_hidden_state shape: (batch_size, seq_len, 768)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Pool token vectors into one sentence vector
        return self.pooling(bert_output.last_hidden_state, attention_mask)

    def forward(self, sentence_a: dict, sentence_b: dict) -> tuple:
        """
        Siamese forward pass — encode both sentences independently.

        Args:
            sentence_a: dict with 'input_ids' and 'attention_mask' for sentence A
            sentence_b: dict with 'input_ids' and 'attention_mask' for sentence B

        Returns:
            (emb_a, emb_b): both of shape (batch_size, 768)
        """
        emb_a = self.encode(sentence_a["input_ids"], sentence_a["attention_mask"])
        emb_b = self.encode(sentence_b["input_ids"], sentence_b["attention_mask"])
        return emb_a, emb_b

    def tokenize(self, sentences: list, max_length: int = 128) -> dict:
        """
        Convert a list of raw strings into tensors ready for encode().

        Args:
            sentences:  ['Hello world', 'Another sentence', ...]
            max_length: truncate anything longer than this

        Returns:
            dict with 'input_ids' and 'attention_mask' tensors
        """
        return self.tokenizer(
            sentences,
            padding=True,       # pad shorter sentences to match the longest in batch
            truncation=True,    # cut sentences longer than max_length
            max_length=max_length,
            return_tensors="pt" # return PyTorch tensors
        )

    def encode_sentences(self, sentences: list, batch_size: int = 64,
                          device: str = "cpu") -> torch.Tensor:
        """
        Convenience method: encode a large list of sentences in batches.
        Useful during evaluation when you have thousands of sentences.

        Returns:
            all_embeddings: (num_sentences, 768) on CPU as numpy-ready tensor
        """
        self.eval()
        all_embeddings = []

        # Process in chunks to avoid running out of GPU memory
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            tokens = self.tokenize(batch, max_length=128)

            # Move to the right device (GPU if available)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():  # no gradients needed during evaluation
                embeddings = self.encode(tokens["input_ids"], tokens["attention_mask"])

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
