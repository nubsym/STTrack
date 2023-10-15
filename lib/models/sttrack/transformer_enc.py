"""
Transformer encoder class
"""
import copy
import random

import torch
import torch.nn.functional as F
from torch import nn

from .box_transformer import get_noise
from .position_encoding import PositionEmbeddingSine
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .transformer import TransformerEncoderLayerLite, TransformerEncoderLite
from ...utils.misc import NestedTensor


class Transformer_enc(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

        self.linear1 = nn.Linear(d_model*2, d_model)
        self.linear2 = nn.Linear(d_model, d_model*2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model*2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, mask=None, pos_embed=None, return_intermediate=False):
        """

        :param feat: (n, bs, C)
        :param mask: (bs, n)
        :param pos_embed: (n, bs, C)
        :param return_intermediate: whether to return intermediate layers' feature maps
        :return:
        """
        # noise = get_noise((1, self.d_model*2), 'gaussian')
        # feat[random.randint(0, len(feat)-1)] = noise.repeat(feat.shape[1], 1)  # 模拟场景中没有目标
        feat_wnoise = get_noise(feat, 'gaussian')
        feat_in = self.linear1(feat_wnoise)
        # feat_in[random.randint(0, len(feat_in) - 1)] = noise.repeat(feat_in.shape[1], 1)
        # feat_in_wnoise = torch.cat(noise.repeat(feat_in.shape[1], 1), dim=-1)
        memory = self.encoder(feat_in, src_key_padding_mask=mask, pos=pos_embed, return_intermediate=return_intermediate)
        feat_out = self.norm(feat_wnoise + self.dropout(self.relu(self.linear2(memory))))
        return feat_out  # (1,b,n,c)[-1].unsqueeze(0).unsqueeze(0).transpose(1, 2)


class Transformer_enc_lite(nn.Module):
    """search feature as queries, concatenated feature as keys and values"""
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        assert num_encoder_layers == 1  # this class is design for encoder with single layer
        encoder_layer = TransformerEncoderLayerLite(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoderLite(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq_dict, return_intermediate=False):
        """search region features as queries, concatenated features as keys and values"""
        memory = self.encoder(seq_dict, return_intermediate=return_intermediate, part_att=self.part_att)
        return memory


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_enc(cfg):
    lite_mode = getattr(cfg.MODEL.TRANSFORMER, "LITE", False)  # queries only use search, rather than concat
    part_att = getattr(cfg.MODEL.TRANSFORMER, "PART_ATT", False)  # keys and values only use template, rather than cat
    if lite_mode:
        print("Building lite encoder...")
        encoder_class = Transformer_enc_lite
    else:
        encoder_class = Transformer_enc
    encoder = encoder_class(
        d_model=cfg.MODEL.HIDDEN_DIM // 2,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRAJECT.ENC_LAYERS,
        normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
        divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM
    )
    encoder.lite_mode = lite_mode
    encoder.part_att = part_att
    return encoder


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
