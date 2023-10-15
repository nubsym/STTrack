import copy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class CFA(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm, num_layers)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_temp, src_search):

        hs, weight = self.decoder(src_temp, src_search)
        return hs, weight


class Decoder(nn.Module):

    def __init__(self, decoderCFA_layer, norm=None, num_layers=4):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, num_layers)
        self.norm = norm
        self.num_layers = num_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos_enc: Optional[Tensor] = None,
                pos_dec: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output, weight = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos_enc=pos_enc, pos_dec=pos_dec)

        if self.norm is not None:
            output = self.norm(output)

        return output, weight


class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos_enc: Optional[Tensor] = None,
                     pos_dec: Optional[Tensor] = None):
        tgt2, weight = self.multihead_attn(query=self.with_pos_embed(tgt, pos_dec),
                                   key=self.with_pos_embed(memory, pos_enc),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt, weight


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_cfa(cfg):
    return CFA(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.CFA.DROPOUT,
        nhead=cfg.MODEL.CFA.NHEADS,
        dim_feedforward=cfg.MODEL.CFA.DIM_FEEDFORWARD,
        num_layers=cfg.MODEL.CFA.LAYERS
    )
