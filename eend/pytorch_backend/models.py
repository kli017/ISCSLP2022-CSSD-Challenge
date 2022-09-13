# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from itertools import permutations
sys.path.append("/home/likai/eend_fastapi/eend/")
from pytorch_backend.transformer import TransformerEncoder
from eend.pytorch_backend.CF import ConformerEncoder
import padertorch as pt



class ConformerDiarization(nn.Module):
    def __init__(self,
                 n_speakers,
                 input_dim,
                 n_units,
                 n_heads,
                 n_layers,
                 dropout_rate,
                 all_n_speakers,
                 d):
        super(ConformerDiarization, self).__init__()
        # self.enc = ConformerEncoder(
        #     input_dim, n_layers, n_units, n_head=n_heads,
        #     ff_dropout=dropout_rate, chunk_size=chunk_size)

        self.enc = ConformerEncoder(
            input_dim, n_layers, n_units, n_head=n_heads, 
            ff_dropout=dropout_rate, mha_dropout=dropout_rate)
        self.linear = nn.Linear(n_units, n_speakers)

        for i in range(n_speakers):
            setattr(self, '{}{:d}'.format("linear", i), nn.Linear(n_units, d))

        self.n_speakers = n_speakers
        self.embed = nn.Embedding(all_n_speakers, d)
        self.alpha = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])
        self.beta = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])

    def modfy_emb(self, weight):
        self.embed = nn.Embedding.from_pretrained(weight)

    def forward(self, xs):
        # Since xs is pre-padded, the following code is extra,
        # but necessary for reproducibility
        xs = nn.utils.rnn.pad_sequence(xs, padding_value=-1, batch_first=True)
        pad_shape = xs.shape
        emb = self.enc(xs)
        ys = self.linear(emb)
        ys = ys.reshape(pad_shape[0], pad_shape[1], -1)

        spksvecs = []
        for i in range(self.n_speakers):
            spkivecs = getattr(self, '{}{:d}'.format("linear", i))(emb)
            spkivecs = spkivecs.reshape(pad_shape[0], pad_shape[1], -1)
            spksvecs.append(spkivecs)

        return ys, spksvecs

    def batch_estimate(self, xs):
        out = self(xs)
        ys = out[0]
        spksvecs = out[1]
        spksvecs = list(zip(*spksvecs))
        outputs = [
                self.estimate(spksvec, y)
                for (spksvec, y) in zip(spksvecs, ys)]
        outputs = list(zip(*outputs))

        return outputs

    def batch_estimate_with_perm(self, xs, ts, ilens=None):
        out = self(xs)
        ys = out[0]
        if ts[0].shape[1] > ys[0].shape[1]:
            # e.g. the case of training 3-spk model with 4-spk data
            add_dim = ts[0].shape[1] - ys[0].shape[1]
            y_device = ys[0].device
            zeros = [torch.zeros(ts[0].shape).to(y_device)
                     for i in range(len(ts))]
            _ys = []
            for zero, y in zip(zeros, ys):
                _zero = zero
                _zero[:, :-add_dim] = y
                _ys.append(_zero)
            _, sigmas = batch_pit_loss(_ys, ts, ilens)
        else:
            _, sigmas = batch_pit_loss(ys, ts, ilens)
        spksvecs = out[1]
        spksvecs = list(zip(*spksvecs))
        outputs = [self.estimate(spksvec, y)
                   for (spksvec, y) in zip(spksvecs, ys)]
        outputs = list(zip(*outputs))
        zs = outputs[0]

        if ts[0].shape[1] > ys[0].shape[1]:
            # e.g. the case of training 3-spk model with 4-spk data
            add_dim = ts[0].shape[1] - ys[0].shape[1]
            z_device = zs[0].device
            zeros = [torch.zeros(ts[0].shape).to(z_device)
                     for i in range(len(ts))]
            _zs = []
            for zero, z in zip(zeros, zs):
                _zero = zero
                _zero[:, :-add_dim] = z
                _zs.append(_zero)
            zs = _zs
            outputs[0] = zs
        outputs.append(sigmas)

        # outputs: [zs, nmz_wavg_spk0vecs, nmz_wavg_spk1vecs, ..., sigmas]
        return outputs

    def estimate(self, spksvec, y):
        outputs = []
        z = torch.sigmoid(y.transpose(1, 0))

        outputs.append(z.transpose(1, 0))
        for spkid, spkvec in enumerate(spksvec):
            norm_spkvec_inv = 1.0 / torch.norm(spkvec, dim=1)
            # Normalize speaker vectors before weighted average
            spkvec = torch.mul(
                    spkvec.transpose(1, 0), norm_spkvec_inv
                    ).transpose(1, 0)
            wavg_spkvec = torch.mul(
                    spkvec.transpose(1, 0), z[spkid]
                    ).transpose(1, 0)
            sum_wavg_spkvec = torch.sum(wavg_spkvec, dim=0)
            nmz_wavg_spkvec = sum_wavg_spkvec / torch.norm(sum_wavg_spkvec)
            outputs.append(nmz_wavg_spkvec)

        # outputs: [z, nmz_wavg_spk0vec, nmz_wavg_spk1vec, ...]
        return outputs
