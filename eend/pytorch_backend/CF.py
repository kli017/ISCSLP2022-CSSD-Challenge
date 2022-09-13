#!/usr/bin/env python3

# Copyright 2020 Masao Someki

import torch
import torch.nn as nn
import numpy as np
import math

from .conformer import FFModule
from .conformer import MHAModule
from .conformer import KMeansMHA
from .conformer import ConvModule
from .conformer import Residual
from .conformer import Conv2dSubsampling


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_model=512,
        ff1_hsize=1024,
        ff1_dropout=0,
        n_head=8,
        mha_dropout=0.1,
        kernel_size=3,
        conv_dropout=0.1,
        ff2_hsize=1024,
        ff2_dropout=0.1,
        batch_size=64,
        max_seq_length=512,
        window_size=128,
        decay=0.999,
        kmeans_dropout=0.1,
        is_left_to_right=False,
        is_share_qk=False,
        use_kmeans_mha=False,
    ):
        """ConformerBlock.
        Args:
            d_model (int): Embedded dimension of input.
            ff1_hsize (int): Hidden size of th first FFN
            ff1_drop (float): Dropout rate for the first FFN
            n_head (int): Number of heads for MHA
            mha_dropout (float): Dropout rate for the first MHA
            epsilon (float): Epsilon
            kernel_size (int): Kernel_size for the Conv
            conv_dropout (float): Dropout rate for the first Conv
            ff2_hsize (int): Hidden size of th first FFN
            ff2_drop (float): Dropout rate for the first FFN
            km_config (dict): Config for KMeans Attention.
            use_kmeans_mha(boolean): Flag to use KMeans Attention for multi-head attention.
        """
        super(ConformerBlock, self).__init__()

        self.ff_module1 = Residual(
            module=FFModule(
                d_model=d_model,
                h_size=ff1_hsize,
                dropout=ff1_dropout
            ),
            half=True
        )
        if use_kmeans_mha:
            self.mha_module = Residual(
                module=KMeansMHA(
                    d_model=d_model,
                    n_head=n_head,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                    window_size=window_size,
                    decay=decay,
                    dropout=kmeans_dropout,
                    is_left_to_right=is_left_to_right,
                    is_share_qk=is_share_qk,
                )
            )
        else:
            self.mha_module = Residual(
                module=MHAModule(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=mha_dropout
                )
            )
        self.conv_module = Residual(
            module=ConvModule(
                in_channels=d_model,
                kernel_size=kernel_size,
                dropout=conv_dropout
            )
        )
        self.ff_module2 = Residual(
            FFModule(
                d_model=d_model,
                h_size=ff2_hsize,
                dropout=ff2_dropout
            ),
            half=True
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, **kwargs):
        """Forward propagation of conformer block.
        Args:
            inputs (torch.Tensor): Input tensor. Shape is [B, L, D]
        Returns:
            torch.Tensor
        """
        x = self.ff_module1(inputs)
        x = self.mha_module(x, **kwargs)
        x = self.conv_module(x)
        x = self.ff_module2(x)
        x = self.layer_norm(x)
        return x

def get_conformer(config):
    return ConformerBlock(
        d_model=config.d_model,
        ff1_hsize=config.ff1_hsize,
        ff1_dropout=config.ff1_dropout,
        n_head=config.n_head,
        mha_dropout=config.mha_dropout,
        kernel_size=config.kernel_size,
        conv_dropout=config.conv_dropout,
        ff2_hsize=config.ff2_hsize,
        ff2_dropout=config.ff2_dropout,
        km_config=config.km_config,
        use_kmeans_mha=config.use_kmeans_mha
    )


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the
        tokens in the sequence. The positional encodings have the same dimension
        as the embeddings, so that the two can be summed. Here, we use sine and
        cosine functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConformerEncoder(nn.Module):
    def __init__(self,
                idim, 
                n_layers, 
                n_units,
                ff_hsize=256,
                ff_dropout=0.1,
                n_head=8, 
                mha_dropout=0.1, 
                kernel_size=11): ## 11
        super(ConformerEncoder, self).__init__()
        self.n_layer = n_layers
        self.linear_in = nn.Linear(idim, n_units)
        self.linear_dropout = nn.Dropout(p=0.1)
        for i in range(n_layers):
            setattr(self, '{}{:d}'.format("conformer_", i),
            ConformerBlock(d_model=n_units, 
                      ff1_hsize=ff_hsize, 
                      ff1_dropout=ff_dropout,
                      n_head=n_head, 
                      mha_dropout=mha_dropout, 
                      kernel_size=kernel_size,
                      ff2_hsize=ff_hsize, 
                      ff2_dropout=ff_dropout))
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        BT_size = x.shape[0] * x.shape[1]
        e = self.linear_in(x)
        e = self.linear_dropout(e)
        for i in range(self.n_layer):
            e = getattr(self, '{}{:d}'.format("conformer_", i))(e)
        
        return self.lnorm_out(e.reshape(BT_size, -1))
