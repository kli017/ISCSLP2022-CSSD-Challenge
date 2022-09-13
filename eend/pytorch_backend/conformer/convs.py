#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Masao Someki

"""ConvolutionModule definition."""

import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


# class ConvModule(nn.Module):
#     """ConvolutionModule in Conformer model.
#     :param int channels: channels of cnn
#     :param int kernel_size: kernerl size of cnn
#     """

#     def __init__(self, in_channels, kernel_size=3, dropout=0.1, bias=False):
#         """Construct an ConvolutionModule object."""
#         super(ConvModule, self).__init__()
#         # kernerl_size should be a odd number for 'SAME' padding
#         assert (kernel_size - 1) % 2 == 0

#         self.layer_norm = nn.LayerNorm(in_channels)

#         self.pointwise_conv1 = nn.Conv1d(
#             in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
#         )
#         self.glu_activation = F.glu

#         self.depthwise_conv = nn.Conv1d(
#             in_channels,
#             in_channels,
#             kernel_size,
#             stride=1,
#             padding=(kernel_size - 1) // 2,
#             groups=in_channels,
#             bias=bias
#         )

#         self.batch_norm = nn.BatchNorm1d(in_channels)

#         self.swish_activation = swish

#         self.pointwise_conv2 = nn.Conv1d(
#             in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
#         )

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         """Compute convolution module.
#         Args:
#             x (Tensor): Shape of x is (Batch, Length, Dim)
#         Returns:
#             Tensor: (B, L, D)
#         """
#         x = self.layer_norm(x)                       # (B, L, D)
#         x = self.pointwise_conv1(x.transpose(1, 2))  # (B, D*2, L)
#         x = self.glu_activation(x, dim=1)            # (B, D, L)
#         x = self.depthwise_conv(x)                   # (B, D, L)
#         x = self.batch_norm(x)                       # (B, D, L)
#         x = self.swish_activation(x)                 # (B, D, L)
#         x = self.pointwise_conv2(x)                  # (B, D, L)
#         x = self.dropout(x)

#         return x.transpose(1, 2)                     # (B, L, D)

class ConvModule(nn.Module):
    """ConvolutionModule in Conformer model.
    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn
    """

    def __init__(self, in_channels, kernel_size=3, dropout=0.2, bias=False):
        """Construct an ConvolutionModule object."""
        super(ConvModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.layer_norm = nn.LayerNorm(in_channels)

        self.pos_conv1 = nn.Conv1d(
            in_channels, 2 * in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.glu_activation = F.glu

        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=in_channels,
            bias=bias
        )

        self.batch_norm = nn.BatchNorm1d(in_channels)

        self.swish_activation = swish

        self.pointwise_conv2 = nn.Conv1d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (Tensor): Shape of x is (Batch, Length, Dim)
        Returns:
            Tensor: (B, L, D)
        """
        x = self.layer_norm(x) # (B, L, D)
        x = self.pos_conv1(x.transpose(1, 2))  # (B, D*2, L)
        x = self.glu_activation(x, dim=1)  # (B, D, L)
        x = self.depthwise_conv(x) # (B, D, L)
        x = self.batch_norm(x) # (B, D, L)
        x = self.swish_activation(x) # (B, D, L)
        x = self.pointwise_conv2(x) # (B, D, L)
        x = self.dropout(x)

        return x.transpose(1, 2) # (B, L, D)

class depthwise_separable_conv(nn.Module):
    """2d depthwise separable convolution"""
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, bias=bias, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Conv2dSubsampling(nn.Module):
    """2d convolution subsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._layers(in_channels, out_channels)

    def _layers(self, in_channels, out_channels):
        #self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=5)
        self.conv1 = depthwise_separable_conv(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = depthwise_separable_conv(out_channels, out_channels, kernel_size=7, stride=5)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.conv1(x.unsqueeze(1)))
        x = torch.relu(self.conv2(x))
        batch_size, channels, subsampled_lengths, subsampled_dim = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)
        # output_lengths = input_lengths >> 2
        # output_lengths -= 1
        return x