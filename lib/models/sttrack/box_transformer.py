import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
# from lib.models.visual import pltshow


class NoiseResNet3x3Conv(nn.Module):
  def __init__(self, channels=32):
    super().__init__()
    self.conv_2d_1 = nn.Conv2d(in_channels=channels,
                               out_channels=20,
                               kernel_size=1,
                               stride=1,
                               padding=0)
    self.conv_2d_2 = nn.Conv2d(in_channels=20,
                               out_channels=20,
                               kernel_size=3,
                               stride=1,
                               padding=0)
    self.conv_2d_3 = nn.Conv2d(in_channels=20,
                               out_channels=20,
                               kernel_size=1,
                               stride=1,
                               padding=0)
    self.conv_2d_4 = nn.Conv2d(in_channels=20,
                               out_channels=channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

  def forward(self, x):
    bs, ch, nx, ny = x.permute(1,2,0).unsqueeze(-1).shape
    x = torch.empty((bs, ch, nx + 2, ny + 2), device=x.device).normal_()
    residual = x[:, :, 1:-1, 1:-1]
    x = F.leaky_relu(self.conv_2d_1(x))
    x = F.leaky_relu(self.conv_2d_2(x))
    x = F.leaky_relu(self.conv_2d_3(x))
    x = self.conv_2d_4(x) + residual
    return x


# star noise generate
def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class BoxTransformer(nn.Module):

    def __init__(self, d_model=32, nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, divide_norm=divide_norm)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        if num_encoder_layers == 0:
            self.encoder = None
        else:
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.d_feed = dim_feedforward
        # 2021.1.7 Try dividing norm to avoid NAN
        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

        self.feat_len = 2
        self.input_embedding_layer_temporal = torch.nn.Linear(self.feat_len, d_model)
        self.output_embedding_layer_temporal = torch.nn.Linear(d_model+16, self.feat_len)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        # self.noise_gen = NoiseResNet3x3Conv(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, trajectory):
        """
        :param trajectory: (f-1,B,2)
        :return:(f_last,B,2)
        """
        noise = get_noise((1, 16), 'gaussian')
        if self.encoder is None:
            outputs_current = trajectory[-1]
        else:
            trajectory_in = self.dropout(self.relu(self.input_embedding_layer_temporal(trajectory)))
            memory = self.encoder(trajectory_in)[-1]
            noise_to_cat = noise.repeat(memory.shape[0], 1)
            temporal_input_embedded_wnoise = torch.cat((memory, noise_to_cat), dim=1)  # 模拟输出多样性
            outputs_current = self.output_embedding_layer_temporal(temporal_input_embedded_wnoise)
        return outputs_current  # (B,4)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_intermediate=False):
        if return_intermediate:
            output_list = []
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)
                if self.norm is None:
                    output_list.append(output)
            if self.norm is not None:
                output = self.norm(output)
                output_list.append(output)
            return output_list
        else:
            output = src

            for layer in self.layers:
                output = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

            if self.norm is not None:
                output = self.norm(output)

            return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, divide_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before  # first normalization, then add

        self.divide_norm = divide_norm
        self.scale_factor = float(d_model // nhead) ** 0.5

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        src2, att = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def build_boxformer(cfg):
    return BoxTransformer(
        d_model=cfg.MODEL.TRAJECT.DMODEL,
        dropout=cfg.MODEL.TRAJECT.DROPOUT,
        nhead=cfg.MODEL.TRAJECT.NHEADS,
        dim_feedforward=cfg.MODEL.TRAJECT.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRAJECT.ENC_LAYERS
    )


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
