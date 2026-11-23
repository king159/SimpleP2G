"""Gene-DML image-path modules, following the pinned upstream module_new.py.

Source: YXSong000/Gene-DML, commit 385790a4f4f3fc041e1cef3eed53e23c0bb4e543.
Parameter names match upstream for numerical comparison. Global positional
convolution gathers back into input spot order instead of returning COO order.
Ordinary PyTorch train/eval semantics replace the upstream attention override.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F


class PreNorm(nn.Module):
    def __init__(self, emb_dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0.0, attn_bias=False, resolution=(5, 5)):
        super().__init__()
        if emb_dim % heads:
            raise ValueError("Feature dimension must be divisible by the number of heads")
        self.heads = heads
        self.scale = (emb_dim // heads) ** -0.5
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Dropout(dropout)) if heads != 1 else nn.Identity()
        self.attn_bias = attn_bias
        if attn_bias:
            points = list(itertools.product(range(resolution[0]), range(resolution[1])))
            offsets, indices = {}, []
            for a in points:
                for b in points:
                    offset = (abs(a[0] - b[0]), abs(a[1] - b[1]))
                    offsets.setdefault(offset, len(offsets))
                    indices.append(offsets[offset])
            self.attention_biases = nn.Parameter(torch.zeros(heads, len(offsets)))
            self.register_buffer(
                "attention_bias_idxs", torch.tensor(indices).view(len(points), len(points)), persistent=False
            )

    def forward(self, x, mask=None):
        batch, length, width = x.shape
        q, k, v = [
            t.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        ]
        if not self.attn_bias and mask is None:
            # The released attention has no dropout on attention weights.
            # SDPA avoids materializing the whole-slide N x N matrix while
            # retaining the same scaling, projections and output dropout.
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
            return self.to_out(out.transpose(1, 2).reshape(batch, length, width))
        # Keep the small 25-token neighbor path's learned relative bias and
        # finite masked-logit convention exactly as in the release.
        scores = (q @ k.transpose(-1, -2)) * self.scale
        if self.attn_bias:
            scores = scores + self.attention_biases[:, self.attention_bias_idxs]
        if mask is not None:
            # Upstream masks keys, retaining every query (including masked grid
            # locations) for the later 25-token mean. Do not change that pooling.
            scores = scores.masked_fill(mask == 0, torch.finfo(torch.float16).min)
        out = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, width)
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.0, attn_bias=False, resolution=(5, 5)):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads, dropout, attn_bias, resolution)),
                        PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None):
        for attention, ff in self.layers:
            x = attention(x, mask=mask) + x
            x = ff(x) + x
        return x


class PEGH(nn.Module):
    def __init__(self, dim=512, kernel_size=3, max_grid_cells=262144):
        super().__init__()
        self.proj1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.max_grid_cells = max_grid_cells

    def forward(self, x, pos):
        if x.ndim != 3 or x.shape[0] != 1 or pos.shape != (x.shape[1], 2):
            raise ValueError("Global input must be (1, N, D) with (N, 2) lattice positions")
        if pos.dtype not in (torch.int32, torch.int64):
            raise ValueError("Global positions must be integer ST lattice coordinates, not WSI pixel coordinates")
        pos = pos - pos.min(dim=0).values
        if pos.unique(dim=0).shape[0] != pos.shape[0]:
            raise ValueError("Duplicate lattice positions would merge distinct spots in positional convolution")
        height, width = (int(value) + 1 for value in pos.max(dim=0).values)
        if height * width > self.max_grid_cells:
            raise ValueError("Global lattice exceeds max_grid_cells; verify that pixel coordinates were not supplied")
        # Match upstream's [D, second coordinate, first coordinate] grid axes.
        flat = pos[:, 1] * height + pos[:, 0]
        dense = x.new_zeros((x.shape[-1], width * height)).index_copy(1, flat, x[0].T)
        dense = dense.reshape(1, x.shape[-1], width, height)
        convolved = self.proj1(dense)
        # Explicit gather avoids silent COO sorting, missing zero-vector spots,
        # and squeeze() failures when a slide contains just one spot.
        return (convolved + dense).flatten(2).index_select(2, flat).transpose(1, 2)


class GlobalEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.0, kernel_size=3, max_grid_cells=262144):
        super().__init__()
        if depth < 1:
            raise ValueError("Global depth must be positive")
        self.pos_layer = PEGH(emb_dim, kernel_size, max_grid_cells)
        self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout)
        self.layer2 = TransformerEncoder(emb_dim, depth - 1, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, position):
        return self.norm(self.layer2(self.pos_layer(self.layer1(x), position)))


class NeighborEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.0, resolution=(5, 5)):
        super().__init__()
        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, True, resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        return self.norm(self.layer(x, mask=None if mask is None else mask[:, None, None, :]))
