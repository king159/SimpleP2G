"""BLEEP baseline

Strictly aligned with https://github.com/bowang-lab/BLEEP implementation architecture,
but enhanced to accept precomputed image embeddings (embedding-mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BleepConfig:
    cond_size: int
    num_genes: int

    proj_dim: int = 256
    # hidden_dim not used in reference architecture
    hidden_dim: int = 512
    dropout: float = 0.1

    # Reference uses fixed temperature scalar (default 1.0)
    temperature: float = 1.0

    # Retrieval settings
    topk: int = 64
    bank_subsample: int = -1
    bank_batch_size: int = 4096


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        # Reference:
        # self.projection = nn.Linear(embedding_dim, projection_dim)
        # self.gelu = nn.GELU()
        # self.fc = nn.Linear(projection_dim, projection_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(projection_dim)

        self.projection = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.drop(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def _soft_cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(preds, dim=-1)
    return (-targets * log_probs).sum(dim=1)


class BleepModel(nn.Module):
    def __init__(self, cfg: BleepConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.temperature <= 0:
            raise ValueError("temperature must be > 0")

        # In enhanced mode, 'cond_size' is the input dimension
        self.image_projection = ProjectionHead(
            in_dim=cfg.cond_size,
            out_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )
        self.spot_projection = ProjectionHead(
            in_dim=cfg.num_genes,
            out_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )

        self.temperature = cfg.temperature

    def encode_image(self, y: torch.Tensor) -> torch.Tensor:
        # y is already the image embedding (from pretrained encoder)
        z = self.image_projection(y)
        # Reference does NOT normalize here
        return z

    def encode_spot(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spot_projection(x)
        # Reference does NOT normalize here
        return z

    def compute_loss(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Contrastive loss between image embeddings (y) and gene vectors (x)."""
        image_z = self.encode_image(y)
        spot_z = self.encode_spot(x)

        # Reference: logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        logits = (spot_z @ image_z.t()) / self.temperature

        images_similarity = image_z @ image_z.t()
        spots_similarity = spot_z @ spot_z.t()

        # Reference: targets = F.softmax(((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1)
        targets = F.softmax(((images_similarity + spots_similarity) / 2.0) / self.temperature, dim=-1)

        spots_loss = _soft_cross_entropy(logits, targets)
        images_loss = _soft_cross_entropy(logits.t(), targets.t())

        loss = (spots_loss + images_loss) / 2.0
        return loss.mean()

    @torch.no_grad()
    def build_image_bank(self, bank_y: torch.Tensor, *, batch_size: int) -> torch.Tensor:
        """Compute image embeddings for a bank in chunks."""
        zs = []
        for i in range(0, bank_y.shape[0], batch_size):
            chunk = bank_y[i : i + batch_size]
            zs.append(self.encode_image(chunk))
        return torch.cat(zs, dim=0)

    @torch.no_grad()
    def predict_from_bank(
        self,
        query_y: torch.Tensor,
        bank_image_z: torch.Tensor,
        bank_x: torch.Tensor,
        *,
        topk: int,
    ) -> torch.Tensor:
        """Predict expression for query embeddings by retrieving from a reference bank."""
        query_z = self.encode_image(query_y)

        # Reference calculation for similarity
        sim = (query_z @ bank_image_z.t()) / self.temperature

        if topk is not None and topk > 0 and topk < sim.shape[1]:
            vals, idx = sim.topk(topk, dim=-1)
            weights = F.softmax(vals, dim=-1)
            bank_x_sel = bank_x.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], -1)
            return (weights.unsqueeze(-1) * bank_x_sel).sum(dim=1)

        weights = F.softmax(sim, dim=-1)
        return weights @ bank_x


def build_bleep_from_omegaconf(args) -> BleepModel:
    bleep_cfg = getattr(args, "bleep", None)

    cond_size = int(getattr(args, "cond_size"))
    num_genes = int(getattr(args, "num_genes", getattr(args, "input_gene_size")))

    cfg = BleepConfig(
        cond_size=cond_size,
        num_genes=num_genes,
        proj_dim=int(getattr(bleep_cfg, "proj_dim", 256)) if bleep_cfg is not None else 256,
        hidden_dim=int(getattr(bleep_cfg, "hidden_dim", 512)) if bleep_cfg is not None else 512,
        dropout=float(getattr(bleep_cfg, "dropout", 0.1)) if bleep_cfg is not None else 0.1,
        temperature=float(getattr(bleep_cfg, "temperature", 1.0)) if bleep_cfg is not None else 1.0,
        topk=int(getattr(bleep_cfg, "topk", 64)) if bleep_cfg is not None else 64,
        bank_subsample=int(getattr(bleep_cfg, "bank_subsample", -1)) if bleep_cfg is not None else -1,
        bank_batch_size=int(getattr(bleep_cfg, "bank_batch_size", 4096)) if bleep_cfg is not None else 4096,
    )
    return BleepModel(cfg)
