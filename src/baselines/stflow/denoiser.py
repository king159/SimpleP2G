"""STFlow denoiser + flow-matching interpolant.

Combines the per-block ``SpatialTransformer`` with sinusoidal time embedding
and a Gaussian-prior flow-matching interpolant. Adapted from
``the released STFlow stflow/model/denoiser.py`` and ``stflow/flow/``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .transformer import SpatialTransformer


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class STFlowDenoiser(nn.Module):
    """STFlow denoiser: image features + noisy gene_exp + time -> predicted gene_exp.

    Parameters
    ----------
    n_genes : int
    feature_dim : int
        encoder embedding dim (e.g. 2048 for ResNet50, 2560 for Virchow2)
    hidden_dim : int
    pairwise_hidden_dim : int
    n_layers : int
    n_heads : int
    dropout : float
    attn_dropout : float
    n_neighbors : int
    activation : str  ("swiglu" | "gelu" | "relu" | "silu")
    """

    def __init__(
        self,
        n_genes: int,
        feature_dim: int,
        hidden_dim: int = 128,
        pairwise_hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.2,
        n_neighbors: int = 8,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.backbone = SpatialTransformer(
            n_genes=n_genes,
            d_input=feature_dim,
            d_model=hidden_dim,
            d_edge_model=pairwise_hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            n_neighbors=n_neighbors,
            activation=activation,
        )
        self.fourier_proj = TimestepEmbedder(hidden_dim)
        self.image_transform = nn.Linear(feature_dim, hidden_dim)
        self.loss_func = nn.MSELoss()

    def inference(self, noisy_exp, img_features, coords, t_steps):
        img_features = self.image_transform(img_features)
        time_emb = self.fourier_proj(t_steps)[:, None].expand(noisy_exp.shape[0], noisy_exp.shape[1], -1)
        features = img_features + time_emb
        return self.backbone(gene_exp=noisy_exp, features=features, coords=coords)

    def forward(self, exp, img_features, coords, labels, t_steps):
        prediction = self.inference(exp, img_features, coords, t_steps)
        pad_mask = img_features.sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        return prediction, loss


class GaussianInterpolant:
    """Linear-interpolant flow matching with a Gaussian prior."""

    def __init__(self):
        pass

    @staticmethod
    def sample_t(shape, device):
        return torch.rand(shape, device=device)

    @staticmethod
    def sample_from_prior(shape, device):
        return torch.randn(shape, device=device)

    def corrupt_exp(self, exp: torch.Tensor):
        """Return ``(x_t, t)`` with x_t = (1-t) * x0 + t * exp,  x0 ~ N(0, I)."""
        device = exp.device
        t = self.sample_t((exp.shape[0],), device)
        exp_0 = self.sample_from_prior(exp.shape, device)
        x_t = exp_0 * (1 - t[:, None, None]) + exp * t[:, None, None]
        return x_t, t

    @staticmethod
    def step(exp_pred, exp_t, t, d_t):
        # vector field: (exp_pred - exp_t) / (1 - t); update by d_t along it
        vf = (exp_pred - exp_t) / (1.0 - t[:, None, None]).clamp_min(1e-6)
        return exp_t + d_t[:, None, None] * vf


class ZINBInterpolant:
    """Linear-interpolant flow matching with a Zero-Inflated Negative-Binomial prior.

    Mirrors ``the released STFlow flow/{interpolant,noise}.py`` with
    ``prior_sample_type='zinb'`` and ``normalize=True`` (i.e. log1p applied to
    the raw ZINB sample). Uses pure torch (no scvi-tools dependency):
    NegativeBinomial(total_count, logits) sample masked by Bernoulli(sigmoid(zi_logits)).
    """

    def __init__(self, total_count: float = 1.0, logits: float = 0.1, zi_logits: float = 0.0):
        self.total_count = float(total_count)
        self.logits = float(logits)
        self.zi_logits = float(zi_logits)

    @staticmethod
    def sample_t(shape, device):
        return torch.rand(shape, device=device)

    def sample_from_prior(self, shape, device):
        # NegativeBinomial parameterised by (total_count, logits)
        nb = torch.distributions.NegativeBinomial(
            total_count=torch.tensor(self.total_count, device=device),
            logits=torch.tensor(self.logits, device=device),
        )
        x = nb.sample(shape)
        zi_prob = torch.sigmoid(torch.tensor(self.zi_logits, device=device))
        mask = torch.bernoulli(torch.full(shape, zi_prob.item(), device=device))
        x = x * (1.0 - mask)  # zero-inflate
        return torch.log1p(x)

    def corrupt_exp(self, exp: torch.Tensor):
        device = exp.device
        t = self.sample_t((exp.shape[0],), device)
        exp_0 = self.sample_from_prior(exp.shape, device).to(exp.dtype)
        x_t = exp_0 * (1 - t[:, None, None]) + exp * t[:, None, None]
        return x_t, t

    @staticmethod
    def step(exp_pred, exp_t, t, d_t):
        vf = (exp_pred - exp_t) / (1.0 - t[:, None, None]).clamp_min(1e-6)
        return exp_t + d_t[:, None, None] * vf
