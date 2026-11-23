"""Gene-DML prediction head over separately extracted features.

Architecture and objective follow YXSong000/Gene-DML at
385790a4f4f3fc041e1cef3eed53e23c0bb4e543 (dml.py + module_new.py).
This is a head-level port; choosing Virchow2, fixed cached target views, common
labels, or a different stopping rule is a benchmark adaptation, not a native
UNI reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .modules import FeedForward, GlobalEncoder, NeighborEncoder, TransformerEncoder


UPSTREAM_COMMIT = "385790a4f4f3fc041e1cef3eed53e23c0bb4e543"


@dataclass(frozen=True)
class GeneDMLCrossConfig:
    # UNI emits 1024-wide tokens; Virchow2 emits 1280-wide tokens, not the
    # 2560-wide [CLS; mean patch] vector used by our old target-only cache.
    image_dim: int = 1024
    gene_dim: int = 200
    projection_dim: int = 512
    neighbor_depth: int = 4
    neighbor_heads: int = 4
    neighbor_mlp_ratio: int = 4
    neighbor_dropout: float = 0.3
    global_depth: int = 5
    global_heads: int = 8
    global_mlp_ratio: int = 4
    global_dropout: float = 0.3
    temperature: float = 0.07
    lambda_cld: float = 0.8
    cld_temperature: float = 0.07
    n_clusters: int = 18
    max_grid_cells: int = 262144
    checkpoint_global: bool = False

    def __post_init__(self):
        if min(self.image_dim, self.gene_dim, self.projection_dim, self.n_clusters) < 1:
            raise ValueError("Dimensions and number of clusters must be positive")
        if min(self.temperature, self.cld_temperature) <= 0:
            raise ValueError("Temperatures must be positive")


class GeneEncoder(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.3):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        return self.layer_norm(self.dropout(self.fc(self.gelu(projected))) + projected)


class Normalize(nn.Module):
    def forward(self, x):
        # Keep upstream's exact L2 normalization for the parity check. Invalid
        # all-zero cluster projections should fail the finite-loss guard.
        return x / x.pow(2).sum(dim=1, keepdim=True).sqrt()


def kmeans(x, clusters, iterations=10):
    """Upstream differentiable centroids; deliberately do NOT detach features."""
    if not 1 <= clusters <= len(x):
        raise ValueError("CLD needs 1 <= n_clusters <= batch size; do not silently skip the clustering loss")
    centers = x[:clusters].clone()
    for _ in range(iterations):
        labels = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(dim=1)
        counts = torch.bincount(labels, minlength=clusters).clamp_min(1).float()
        centers = x.new_zeros((clusters, x.shape[1])).scatter_add(0, labels[:, None].expand_as(x), x)
        centers = centers / counts[:, None]
    return labels, centers


def soft_contrastive_loss(image, gene, temperature):
    logits = (gene @ image.T) / temperature
    targets = (((image @ image.T + gene @ gene.T) / 2) / temperature).softmax(dim=-1)
    # Targets remain differentiable, as in the released objective.
    gene_loss = -(targets * logits.log_softmax(dim=-1)).sum(dim=1)
    image_loss = -(targets.T * logits.T.log_softmax(dim=-1)).sum(dim=1)
    return ((gene_loss + image_loss) / 2).mean()


class ThreePathImageEncoder(nn.Module):
    """Trainable image-context modules after the frozen image backbone."""

    def __init__(self, cfg):
        super().__init__()
        self.checkpoint_global = cfg.checkpoint_global
        self.fc_layer = nn.Linear(cfg.image_dim, cfg.projection_dim)
        self.neighbor_encoder = NeighborEncoder(
            cfg.image_dim,
            cfg.neighbor_depth,
            cfg.neighbor_heads,
            cfg.image_dim * cfg.neighbor_mlp_ratio,
            cfg.neighbor_dropout,
        )
        self.global_encoder = GlobalEncoder(
            cfg.image_dim,
            cfg.global_depth,
            cfg.global_heads,
            cfg.image_dim * cfg.global_mlp_ratio,
            cfg.global_dropout,
            max_grid_cells=cfg.max_grid_cells,
        )

    def _encode_global(self, features, positions):
        if self.checkpoint_global and self.training and torch.is_grad_enabled():
            # Non-reentrant mode supports frozen image inputs while preserving
            # gradients of all trainable global-encoder parameters. RNG replay
            # keeps dropout identical to the ordinary forward/backward path.
            return checkpoint(self.global_encoder, features, positions, use_reentrant=False, preserve_rng_state=True)
        return self.global_encoder(features, positions)

    def forward(
        self,
        target_tokens,
        neighbor_features,
        neighbor_mask,
        global_features,
        positions,
        spot_indices,
        slide_indices=None,
    ):
        neighbor_tokens = self.neighbor_encoder(neighbor_features, neighbor_mask)
        if slide_indices is None:
            global_tokens = self._encode_global(global_features.unsqueeze(0), positions)[0][spot_indices]
        else:
            global_tokens = target_tokens.new_zeros((len(target_tokens), target_tokens.shape[-1]))
            for slide in slide_indices.unique(sorted=True).tolist():
                selected = (slide_indices == slide).nonzero().flatten()
                full_slide = self._encode_global(global_features[slide].unsqueeze(0), positions[slide])[0]
                # Match the released mixed-slide path's FP32 destination even
                # when the trainable global encoder runs under autocast.
                global_tokens = global_tokens.index_copy(
                    0, selected, full_slide[spot_indices[selected]].to(global_tokens.dtype)
                )
        return (self.fc_layer(target_tokens), self.fc_layer(neighbor_tokens), self.fc_layer(global_tokens))


class GeneDMLCross(nn.Module):
    """Single/mixed-slide interface; prediction never accepts gene-label context.

    ``target_tokens``: (B, T, D), retaining the backbone's complete token sequence.
    ``neighbor_features``: (B, 25, D), from an actual row-major 5x5 image grid.
    ``global_features``: (N, D), one pooled image descriptor per slide spot.
    ``positions``: (N, 2) integer ST lattice positions; ``spot_indices``: (B,).
    Labels are optional and used only to return a training objective. The
    image-only ``predictions['fusion']`` output is always the evaluation output.
    For mixed-slide training, pass lists of global_features and positions plus
    (B,) slide_indices indexing those lists; loss negatives span the full batch.
    """

    def __init__(self, config: GeneDMLCrossConfig):
        super().__init__()
        self.config = config
        dim, genes = config.projection_dim, config.gene_dim
        self.image_encoder = ThreePathImageEncoder(config)
        self.gene_encoder = nn.Sequential(GeneEncoder(genes, dim), FeedForward(dim, dim * 2, 0.1))
        self.trans_encoder = nn.Sequential(TransformerEncoder(dim, 1, 1, dim * 2, 0.1), FeedForward(dim, dim * 2, 0.1))
        for name in ("target", "neighbor", "global"):
            setattr(
                self,
                name + "_predictor",
                nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dim, genes)),
            )
        self.fusion_predictor = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, genes),
        )
        self.feature_cluster = nn.Sequential(nn.Linear(dim, dim), Normalize())

    def _validate(
        self, target_tokens, neighbor_features, neighbor_mask, global_features, positions, spot_indices, slide_indices
    ):
        cfg = self.config
        if target_tokens.ndim != 3 or target_tokens.shape[1] < 2 or target_tokens.shape[2] != cfg.image_dim:
            raise ValueError(
                "Gene-DML requires (B, T>=2, image_dim) target tokens; pooled spot embeddings are insufficient"
            )
        batch = len(target_tokens)
        if neighbor_features.shape != (batch, 25, cfg.image_dim):
            raise ValueError("Neighbors must be 25 row-major image-grid features, not K24 nearest spots")
        if neighbor_mask.shape != (batch, 25) or not neighbor_mask.bool().any(dim=-1).all():
            raise ValueError("Neighbor mask must have shape (B, 25), with a valid key in every sample")
        if spot_indices.shape != (batch,) or spot_indices.dtype != torch.long:
            raise ValueError("spot_indices must be int64 indices with shape (B,)")
        if slide_indices is None:
            globals_, positions_ = [global_features], [positions]
            assignments = torch.zeros_like(spot_indices)
        else:
            if not isinstance(global_features, (tuple, list)) or not isinstance(positions, (tuple, list)):
                raise ValueError("Mixed-slide inputs need lists of global features and lattice positions")
            if len(global_features) != len(positions):
                raise ValueError("Mixed-slide global feature and position lists differ in length")
            if slide_indices.shape != (batch,) or slide_indices.dtype != torch.long:
                raise ValueError("slide_indices must be int64 with shape (B,)")
            if int(slide_indices.min()) < 0 or int(slide_indices.max()) >= len(global_features):
                raise ValueError("slide_indices is outside the global feature list")
            globals_, positions_, assignments = global_features, positions, slide_indices
        for index, (features, pos) in enumerate(zip(globals_, positions_)):
            if features.ndim != 2 or features.shape[1] != cfg.image_dim:
                raise ValueError("Global image features must have shape (N, image_dim)")
            if pos.shape != (len(features), 2):
                raise ValueError("Global lattice positions do not align with image features")
            spots = spot_indices[assignments == index]
            if len(spots) and (int(spots.min()) < 0 or int(spots.max()) >= len(features)):
                raise ValueError("Target spot index is outside the supplied slide")

    def forward(
        self,
        target_tokens,
        neighbor_features,
        neighbor_mask,
        global_features,
        positions,
        spot_indices,
        expression=None,
        slide_indices=None,
    ):
        self._validate(
            target_tokens, neighbor_features, neighbor_mask, global_features, positions, spot_indices, slide_indices
        )
        target, neighbor, global_ = self.image_encoder(
            target_tokens,
            neighbor_features,
            neighbor_mask,
            global_features,
            positions,
            spot_indices,
            slide_indices,
        )
        if expression is not None:
            if expression.shape != (len(target), self.config.gene_dim):
                raise ValueError("Expression must be (B, gene_dim), aligned with target spot_indices")
            gene = self.gene_encoder(expression)
        image_tokens = self.trans_encoder(torch.cat((target, neighbor, global_[:, None, :]), dim=1))
        target_length = target.shape[1]
        branches = {
            "target": F.normalize(image_tokens[:, :target_length].mean(dim=1), dim=-1),
            "neighbor": F.normalize(image_tokens[:, target_length:-1].mean(dim=1), dim=-1),
            "global": F.normalize(image_tokens[:, -1], dim=-1),
        }
        image_mean = image_tokens.mean(dim=1)
        predictions = {name: getattr(self, name + "_predictor")(features) for name, features in branches.items()}
        predictions["fusion"] = self.fusion_predictor(image_mean)
        result = {"predictions": predictions}
        if expression is None:
            return result

        gene = F.normalize(gene, dim=-1)
        cfg = self.config
        contrastive = sum(soft_contrastive_loss(features, gene, cfg.temperature) for features in branches.values()) / 3
        prediction_losses = {name + "_pred_loss": F.mse_loss(pred, expression) for name, pred in predictions.items()}
        image_labels, image_centers = kmeans(self.feature_cluster(image_mean), cfg.n_clusters)
        gene_labels, gene_centers = kmeans(self.feature_cluster(gene), cfg.n_clusters)
        cld = (
            F.cross_entropy(image_mean @ gene_centers.T / cfg.cld_temperature, gene_labels)
            + F.cross_entropy(gene @ image_centers.T / cfg.cld_temperature, image_labels)
        ) / 2
        # Upstream computes auxiliary predictor MSEs but optimizes ONLY fusion
        # MSE + the averaged three contrastive losses + lambda * CLD.
        prediction = prediction_losses["fusion_pred_loss"]
        result.update(prediction_losses)
        result.update(contrastive_loss=contrastive, prediction_loss=prediction, CLD_loss=cld)
        result["loss"] = contrastive + prediction + cfg.lambda_cld * cld
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError("Gene-DML produced a non-finite loss; check features and cluster projections")
        return result
