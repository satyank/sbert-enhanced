# this class defines the SBERT model and its key functions 

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from models.pooling import PoolingLayer


class SentenceBERT(nn.Module):

    def __init__(self, model_name: str = "bert-base-uncased", pooling_strategy: str = "mean"):
        super().__init__()

        # loading pre-trained model "bert-base-uncased"
        print(f"Loading BERT model: {model_name}")
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768
        self.pooling = PoolingLayer(hidden_size=self.hidden_size, pooling_strategy=pooling_strategy,)
        self.pooling_strategy = pooling_strategy
        print(f"Pooling strategy: {pooling_strategy}")

    # this method converts a tokenized input into one single embedding
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.pooling(bert_output.last_hidden_state, attention_mask)

    # this method encodes the two given sentnecs independently
    def forward(self, sentence_a: dict, sentence_b: dict) -> tuple:
        emb_a = self.encode(sentence_a["input_ids"], sentence_a["attention_mask"])
        emb_b = self.encode(sentence_b["input_ids"], sentence_b["attention_mask"])
        return emb_a, emb_b

    # this method will convert a list of strings into tensors
    def tokenize(self, sentences: list, max_length: int = 128):
        return self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )


    # encode list of sentences in batches to avoid memory overflow issues
    def encode_sentences(self, sentences: list, batch_size: int = 64, device: str = "cpu"):
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i: i + batch_size]
            tokens = self.tokenize(batch, max_length=128)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.no_grad():
                embeddings = self.encode(tokens["input_ids"], tokens["attention_mask"])
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
