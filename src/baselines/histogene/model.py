"""HisToGene baseline – faithful to the original cross-spot attention design.

Original HisToGene (Pang & Zuo, 2021):
  - Input  : per-slide set of spots, each represented as a flattened image patch.
  - Mechanism: Transformer self-attention operates over **all spots in a slide**
    simultaneously (sequence = spots).  Each spot's prediction is informed by
    every other spot via spatial positional encoding.
  - Output : per-spot gene expression.

**Enhanced mode** replaces the raw-patch input with precomputed foundation-model
embeddings while keeping the architecture otherwise identical:
  - ``nn.Linear(cond_size, dim)`` replaces ``nn.Linear(patch_dim, dim)``
  - Everything else (cross-spot attention, positional encoding, gene head) is
    shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class HisToGeneConfig:
    """Shared config for both original and enhanced HisToGene."""

    num_genes: int

    # Input (enhanced mode)
    cond_size: int | None = None  # foundation-model embedding dim; None for original

    # Input (original mode)
    patch_size: int = 112  # original HisToGene uses r=56 -> 112x112

    # Architecture — match original HisToGene defaults
    dim: int = 1024
    n_layers: int = 4
    heads: int = 16
    dim_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1
    n_pos: int = 64  # grid bins for positional encoding


# ── Building blocks (identical to original HisToGene) ──────────────────────


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, h, -1).permute(0, 2, 1, 3) for t in qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ── Unified HisToGene model ───────────────────────────────────────────────


class HisToGeneModel(nn.Module):
    """Slide-level HisToGene with cross-spot Transformer attention.

    Works in two modes controlled by ``cfg.cond_size``:
      * **Enhanced** (``cond_size is not None``):
        input is ``(1, N_spots, cond_size)`` precomputed embeddings.
      * **Original** (``cond_size is None``):
        input is ``(1, N_spots, 3*patch_size*patch_size)`` flattened raw patches.

    Both share the same Transformer + positional encoding + gene_head.
    """

    def __init__(self, cfg: HisToGeneConfig):
        super().__init__()
        self.cfg = cfg

        # Input projection (the **only** part that differs between modes)
        if cfg.cond_size is not None:
            self.patch_embedding = nn.Linear(cfg.cond_size, cfg.dim)
        else:
            patch_dim = 3 * cfg.patch_size * cfg.patch_size
            self.patch_embedding = nn.Linear(patch_dim, cfg.dim)

        # Spatial positional encoding (integer grid bins)
        self.x_embed = nn.Embedding(cfg.n_pos, cfg.dim)
        self.y_embed = nn.Embedding(cfg.n_pos, cfg.dim)

        self.emb_dropout = nn.Dropout(cfg.emb_dropout)

        # Transformer
        mlp_dim = 2 * cfg.dim  # original HisToGene: mlp_dim = 2 * dim
        self.transformer = Transformer(
            dim=cfg.dim,
            depth=cfg.n_layers,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_dim=mlp_dim,
            dropout=cfg.dropout,
        )

        # Per-spot gene prediction head (NO pooling — each spot outputs independently)
        self.gene_head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.num_genes),
        )

    def forward(
        self,
        features: torch.Tensor,
        pos_bins: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features : (B, N_spots, D) – embeddings or flattened patches
            pos_bins : (B, N_spots, 2) – integer grid positions in [0, n_pos)

        Returns:
            pred : (B, N_spots, num_genes)
        """
        # Project to dim
        x = self.patch_embedding(features)  # (B, N, dim)

        # Add spatial positional encoding
        cx = self.x_embed(pos_bins[:, :, 0])  # (B, N, dim)
        cy = self.y_embed(pos_bins[:, :, 1])  # (B, N, dim)
        x = x + cx + cy

        x = self.emb_dropout(x)

        # Cross-spot Transformer attention
        h = self.transformer(x)  # (B, N, dim)

        # Per-spot output
        out = self.gene_head(h)  # (B, N, num_genes)
        return out

    def compute_loss(
        self,
        features: torch.Tensor,
        pos_bins: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features : (B, N, D)
            pos_bins : (B, N, 2)
            labels   : (B, N, G)
        """
        pred = self.forward(features, pos_bins)
        return F.mse_loss(pred, labels)


# ── Builders ─────────────────────────────────────────────────────────────


def build_histogene_from_omegaconf(args) -> HisToGeneModel:
    """Build for **enhanced** mode (precomputed embeddings)."""
    hcfg = getattr(args, "histogene", None)
    num_genes = int(getattr(args, "num_genes", getattr(args, "input_gene_size", 200)))
    cond_size = int(getattr(args, "cond_size"))

    cfg = HisToGeneConfig(
        num_genes=num_genes,
        cond_size=cond_size,
        dim=int(getattr(hcfg, "dim", 1024)) if hcfg else 1024,
        n_layers=int(getattr(hcfg, "n_layers", 4)) if hcfg else 4,
        heads=int(getattr(hcfg, "heads", 16)) if hcfg else 16,
        dim_head=int(getattr(hcfg, "dim_head", 64)) if hcfg else 64,
        dropout=float(getattr(hcfg, "dropout", 0.1)) if hcfg else 0.1,
        emb_dropout=float(getattr(hcfg, "emb_dropout", 0.1)) if hcfg else 0.1,
        n_pos=int(getattr(hcfg, "n_pos", 64)) if hcfg else 64,
    )
    return HisToGeneModel(cfg)


def build_histogene_original_from_omegaconf(args) -> HisToGeneModel:
    """Build for **original** mode (raw image patches)."""
    hcfg = getattr(args, "histogene", None)
    num_genes = int(getattr(args, "num_genes", getattr(args, "input_gene_size", 200)))

    cfg = HisToGeneConfig(
        num_genes=num_genes,
        cond_size=None,
        patch_size=int(getattr(hcfg, "patch_size", 112)) if hcfg else 112,
        dim=int(getattr(hcfg, "dim", 1024)) if hcfg else 1024,
        n_layers=int(getattr(hcfg, "n_layers", 4)) if hcfg else 4,
        heads=int(getattr(hcfg, "heads", 16)) if hcfg else 16,
        dim_head=int(getattr(hcfg, "dim_head", 64)) if hcfg else 64,
        dropout=float(getattr(hcfg, "dropout", 0.1)) if hcfg else 0.1,
        emb_dropout=float(getattr(hcfg, "emb_dropout", 0.1)) if hcfg else 0.1,
        n_pos=int(getattr(hcfg, "n_pos", 64)) if hcfg else 64,
    )
    return HisToGeneModel(cfg)
