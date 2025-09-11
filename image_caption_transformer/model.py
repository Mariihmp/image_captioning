import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import math
from config import Config


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        self.model = nn.Sequential(*list(efficientnet.children())[:-2])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_features = self.model(image)
        # Reshape from (batch, 1280, 8, 8) to (batch, 64, 1280)
        image_features = image_features.flatten(start_dim=2).permute(0, 2, 1)
        return image_features


class PositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        position = torch.arange(0, config.maxlen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2)
                             * (-math.log(10000.0) / config.d_model))
        pe = torch.zeros(config.maxlen, config.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encodings', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.positional_encodings[:, :seq_len, :]
        return x


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads_encoder, batch_first=True), num_layers=1
        )
        self.ff_1 = nn.Linear(1280, 2048)
        self.ff_2 = nn.Linear(2048, config.d_model)
        self.layer_norm_1 = nn.LayerNorm(1280)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)

    def forward(self, image_features: torch.Tensor):
        image_features = self.layer_norm_1(image_features)
        proj = F.relu(self.ff_1(image_features))
        proj = self.layer_norm_2(F.relu(self.ff_2(proj)))
        return self.encoder(proj)


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.d_model)
        self.pos_emb = PositionalEncoding(config)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.d_model, nhead=config.n_heads_decoder, batch_first=True), num_layers=2
        )
        self.out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, encoder_outputs, tgt_attention_mask=None, tgt_key_padding_mask=None):
        x = self.pos_emb(self.embedding(x))
        decoder_out = self.decoder(
            tgt=x,
            memory=encoder_outputs,
            tgt_mask=tgt_attention_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        preds = self.out(decoder_out)
        return preds
