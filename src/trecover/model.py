import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        x = position * div_term

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.shape[0], :]

        return self.dropout(x)


class TRecover(nn.Module):
    def __init__(self,
                 token_size: int,
                 pe_max_len: int,
                 num_layers: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float):
        assert d_model % n_heads == 0, 'd_model size must be evenly divisible by n_heads size'

        super(TRecover, self).__init__()

        self.token_size = token_size
        self.pe_max_len = pe_max_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.scale = math.sqrt(d_model)

        self.mapping = nn.Linear(in_features=token_size, out_features=d_model)

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=pe_max_len)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                        dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                                        dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)

        self.inv_mapping = nn.Linear(in_features=d_model, out_features=token_size)

        self.params_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.init_weights()

    def __repr__(self) -> str:
        return self.__str__()  # but standard implementation is better

    def __str__(self) -> str:
        return f'TRecover(token_size={self.token_size}, pe_max_len={self.pe_max_len}, num_layers={self.num_layers}, ' \
               f'd_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}, dropout={self.dropout})'

    def init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: Tensor, src_pad_mask: Optional[Tensor]) -> Tensor:
        src = src.transpose(0, 1)

        src = F.relu(self.mapping(src)) * self.scale
        src = self.pe(src)

        return self.encoder(src=src, src_key_padding_mask=src_pad_mask)

    def decode(self,
               tgt_inp: Tensor,
               encoded_src: Tensor,
               tgt_attn_mask: Optional[Tensor],
               tgt_pad_mask: Optional[Tensor],
               src_pad_mask: Optional[Tensor]
               ) -> Tensor:
        tgt_inp = tgt_inp.transpose(0, 1)

        tgt_inp = F.relu(self.mapping(tgt_inp)) * self.scale
        tgt_inp = self.pe(tgt_inp)

        return self.decoder(tgt=tgt_inp, memory=encoded_src, tgt_mask=tgt_attn_mask,
                            tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)

    def predict(self,
                tgt_inp: Tensor,
                encoded_src: Tensor,
                tgt_attn_mask: Optional[Tensor],
                tgt_pad_mask: Optional[Tensor],
                src_pad_mask: Optional[Tensor]
                ) -> Tensor:
        decoded = self.decode(tgt_inp, encoded_src, tgt_attn_mask, tgt_pad_mask, src_pad_mask)

        return self.inv_mapping(decoded).transpose(0, 1)

    def forward(self, src: Tensor,
                src_pad_mask: Optional[Tensor],
                tgt_inp: Tensor,
                tgt_attn_mask: Optional[Tensor],
                tgt_pad_mask: Optional[Tensor]
                ) -> Tensor:
        encoded_src = self.encode(src, src_pad_mask)

        return self.predict(tgt_inp, encoded_src, tgt_attn_mask, tgt_pad_mask, src_pad_mask)

    def save(self, filename: Path) -> None:
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename: Path, device: torch.device, strict: bool = True) -> None:
        self.load_state_dict(torch.load(filename, map_location=device), strict=strict)
