"""The three captioning models, all reading the same (batch, 64, 1280) image features.

Every model has the same interface:
    memory = model.encode(features)
    logits, attention = model.decode(memory, tokens)   # teacher forcing
`attention` is the per-word attention over the 64 image regions, or None.
"""

import math

import torch
from torch import nn

from .features import FEATURE_DIM


class MergeLSTM(nn.Module):
    """CNN-LSTM "merge" baseline: a PyTorch port of the Keras model in 1-baseline_encoder_decoder.

    The image is reduced to one vector (no attention) and added to the LSTM output.
    """

    def __init__(self, vocab_size, feat_dim=FEATURE_DIM, hidden=256, dropout=0.5):
        super().__init__()
        self.image_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, hidden), nn.ReLU())
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, vocab_size))

    def encode(self, features):
        return self.image_proj(features.mean(dim=1))

    def decode(self, memory, tokens):
        hidden, _ = self.lstm(self.dropout(self.embedding(tokens)))
        return self.head(hidden + memory.unsqueeze(1)), None


class SoftAttentionLSTM(nn.Module):
    """LSTM with soft attention over image regions (Xu et al., 2015, "Show, Attend and Tell")."""

    def __init__(self, vocab_size, feat_dim=FEATURE_DIM, embed_dim=512, hidden=512, attn_dim=512, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.att_features = nn.Linear(feat_dim, attn_dim)
        self.att_hidden = nn.Linear(hidden, attn_dim)
        self.att_score = nn.Linear(attn_dim, 1)
        self.gate = nn.Linear(hidden, feat_dim)
        self.init_h = nn.Linear(feat_dim, hidden)
        self.init_c = nn.Linear(feat_dim, hidden)
        self.cell = nn.LSTMCell(embed_dim + feat_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def encode(self, features):
        return features

    def decode(self, memory, tokens):
        mean = memory.mean(dim=1)
        h, c = torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))
        projected = self.att_features(memory)  # (batch, regions, attn_dim)
        embedded = self.embedding(tokens)

        logits, alphas = [], []
        for t in range(tokens.size(1)):
            scores = self.att_score(torch.tanh(projected + self.att_hidden(h).unsqueeze(1))).squeeze(2)
            alpha = scores.softmax(dim=1)
            context = torch.sigmoid(self.gate(h)) * (memory * alpha.unsqueeze(2)).sum(dim=1)
            h, c = self.cell(torch.cat([embedded[:, t], context], dim=1), (h, c))
            logits.append(self.out(self.dropout(h)))
            alphas.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerCaptioner(nn.Module):
    """Transformer encoder-decoder, same design as image_caption_transformer/model.py."""

    def __init__(self, vocab_size, feat_dim=FEATURE_DIM, d_model=512, enc_heads=1, dec_heads=2,
                 enc_layers=1, dec_layers=2, max_len=64, dropout=0.1):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, 2048), nn.ReLU(),
            nn.Linear(2048, d_model), nn.ReLU(), nn.LayerNorm(d_model),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, enc_heads, dropout=dropout, batch_first=True),
            num_layers=enc_layers, enable_nested_tensor=False,
        )
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, dec_heads, dropout=dropout, batch_first=True),
            num_layers=dec_layers,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def encode(self, features):
        return self.encoder(self.image_proj(features))

    def decode(self, memory, tokens):
        length = tokens.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.decoder(
            self.pos(self.embedding(tokens)), memory,
            tgt_mask=causal, tgt_key_padding_mask=tokens.eq(0),
        )
        return self.out(hidden), None


MODELS = {
    "lstm": MergeLSTM,
    "attention": SoftAttentionLSTM,
    "transformer": TransformerCaptioner,
}


def build_model(name, vocab_size, **kwargs):
    return MODELS[name](vocab_size, **kwargs)
