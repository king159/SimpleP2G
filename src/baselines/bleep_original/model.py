from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BleepOriginalConfig:
    num_genes: int

    # Image encoder (timm)
    image_model_name: str = "resnet50"
    image_pretrained: bool = True
    image_trainable: bool = True

    proj_dim: int = 256
    # hidden_dim is not used in reference architecture but kept for config compatibility if needed
    hidden_dim: int = 512
    dropout: float = 0.1
    temperature: float = 1.0  # Reference default is 1.0


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
    # Reference:
    # log_softmax = nn.LogSoftmax(dim=-1)
    # loss = (-targets * log_softmax(preds)).sum(1)

    log_probs = F.log_softmax(preds, dim=-1)
    return (-targets * log_probs).sum(dim=1)


class BleepOriginalModel(nn.Module):
    """Original-modality BLEEP: patch pixels + gene vector CLIP-style loss.
    Strictly aligned with https://github.com/bowang-lab/BLEEP implementation.
    """

    def __init__(self, cfg: BleepOriginalConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.temperature <= 0:
            raise ValueError("temperature must be > 0")

        import timm

        self.image_encoder = timm.create_model(
            cfg.image_model_name,
            pretrained=cfg.image_pretrained,
            num_classes=0,
            global_pool="avg",
        )
        for p in self.image_encoder.parameters():
            p.requires_grad = bool(cfg.image_trainable)

        image_dim = self.image_encoder.num_features

        self.image_projection = ProjectionHead(
            in_dim=int(image_dim),
            out_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )
        self.spot_projection = ProjectionHead(
            in_dim=cfg.num_genes,
            out_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )

        # Reference uses fixed temperature scalar
        self.temperature = cfg.temperature

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        z = self.image_encoder(image)
        z = self.image_projection(z)
        # Reference does NOT normalize here
        return z

    def encode_spot(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spot_projection(x)
        # Reference does NOT normalize here
        return z

    def compute_loss(self, image: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Getting Image and Spot Embeddings (with same dimension)
        image_embeddings = self.encode_image(image)
        spot_embeddings = self.encode_spot(x)

        # Calculating the Loss
        # Reference: logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        logits = (spot_embeddings @ image_embeddings.t()) / self.temperature

        images_similarity = image_embeddings @ image_embeddings.t()
        spots_similarity = spot_embeddings @ spot_embeddings.t()

        # Reference: targets = F.softmax(((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1)
        targets = F.softmax(((images_similarity + spots_similarity) / 2.0) / self.temperature, dim=-1)

        # Reference: spots_loss = cross_entropy(logits, targets, reduction="none")
        # Reference: images_loss = cross_entropy(logits.T, targets.T, reduction="none")
        spots_loss = _soft_cross_entropy(logits, targets)
        images_loss = _soft_cross_entropy(logits.t(), targets.t())

        # Reference: loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        # return loss.mean()
        return ((spots_loss + images_loss) / 2.0).mean()

    @torch.no_grad()
    def build_image_bank(self, image_loader) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a bank from a DataLoader over the training set.

        Returns:
          - bank_image_z: (N, D) float32 on the same device as model
          - bank_x: (N, G) float32 on the same device as model
        """
        self.eval()
        zs = []
        xs = []
        for x, y in image_loader:
            x = x.to(next(self.parameters()).device)
            image = y["image"].to(next(self.parameters()).device)
            zs.append(self.encode_image(image))
            xs.append(x)
        return torch.cat(zs, dim=0), torch.cat(xs, dim=0)

    @torch.no_grad()
    def predict_from_bank(self, image: torch.Tensor, bank_image_z: torch.Tensor, bank_x: torch.Tensor, *, topk: int):
        self.eval()
        query_z = self.encode_image(image)
        # Reference calculation for similarity
        sim = (query_z @ bank_image_z.t()) / self.temperature

        if topk is not None and topk > 0 and topk < sim.shape[1]:
            vals, idx = sim.topk(topk, dim=-1)
            weights = F.softmax(vals, dim=-1)
            bank_x_sel = bank_x.index_select(0, idx.reshape(-1)).reshape(idx.shape[0], idx.shape[1], -1)
            return (weights.unsqueeze(-1) * bank_x_sel).sum(dim=1)

        weights = F.softmax(sim, dim=-1)
        return weights @ bank_x


def build_bleep_original_from_omegaconf(args) -> BleepOriginalModel:
    bleep_cfg = getattr(args, "bleep", None)
    num_genes = int(getattr(args, "num_genes", getattr(args, "input_gene_size")))

    cfg = BleepOriginalConfig(
        num_genes=num_genes,
        image_model_name=str(getattr(bleep_cfg, "image_model_name", "resnet50"))
        if bleep_cfg is not None
        else "resnet50",
        image_pretrained=bool(getattr(bleep_cfg, "image_pretrained", True)) if bleep_cfg is not None else True,
        image_trainable=bool(getattr(bleep_cfg, "image_trainable", True)) if bleep_cfg is not None else True,
        proj_dim=int(getattr(bleep_cfg, "proj_dim", 256)) if bleep_cfg is not None else 256,
        hidden_dim=int(getattr(bleep_cfg, "hidden_dim", 512)) if bleep_cfg is not None else 512,
        dropout=float(getattr(bleep_cfg, "dropout", 0.1)) if bleep_cfg is not None else 0.1,
        temperature=float(getattr(bleep_cfg, "temperature", 1.0)) if bleep_cfg is not None else 1.0,
    )
    return BleepOriginalModel(cfg)
