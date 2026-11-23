"""Train / eval API for the STFlow baseline.

Uses slide-level batching (batch_size=1). Each batch is the dict produced by
``SlideEmbeddingDataset.__getitem__``. We add a leading B=1 dim to align with
the (B, N, *) tensor shapes the SpatialTransformer expects.
"""

from __future__ import annotations

import numpy as np
import torch

from src.baselines.stflow.denoiser import (
    GaussianInterpolant,
    STFlowDenoiser,
    ZINBInterpolant,
)


def build_stflow_from_omegaconf(args):
    return STFlowDenoiser(
        n_genes=int(args.input_gene_size),
        feature_dim=int(args.cond_size),
        hidden_dim=int(getattr(args, "hidden_dim", 128)),
        pairwise_hidden_dim=int(getattr(args, "pairwise_hidden_dim", 128)),
        n_layers=int(getattr(args, "n_layers", 4)),
        n_heads=int(getattr(args, "n_heads", 4)),
        dropout=float(getattr(args, "dropout", 0.2)),
        attn_dropout=float(getattr(args, "attn_dropout", 0.2)),
        n_neighbors=int(getattr(args, "n_neighbors", 8)),
        activation=str(getattr(args, "activation", "gelu")),
    )


def build_interpolant_from_omegaconf(args):
    """Build interpolant from config. Defaults to ZINB (paper-faithful)."""
    prior = str(getattr(args, "prior_sampler", "zinb")).lower()
    if prior == "gaussian":
        return GaussianInterpolant()
    if prior == "zinb":
        return ZINBInterpolant(
            total_count=float(getattr(args, "zinb_total_count", 1.0)),
            logits=float(getattr(args, "zinb_logits", 0.1)),
            zi_logits=float(getattr(args, "zinb_zi_logits", 0.0)),
        )
    raise ValueError(f"Unknown prior_sampler={prior!r}")


def _to_3d(t: torch.Tensor) -> torch.Tensor:
    """Ensure leading batch dim. DataLoader with batch_size=1 already adds one."""
    if t.dim() == 2:
        return t.unsqueeze(0)
    return t


def train_one_step(model, batch, args, **kwargs):
    interpolant = kwargs["interpolant"]
    embeddings = _to_3d(batch["embeddings"])  # (1, N, D)
    coords = _to_3d(batch["coords"])  # (1, N, 2)
    labels = _to_3d(batch["labels"])  # (1, N, G)

    noisy_exp, t_steps = interpolant.corrupt_exp(labels)
    _pred, loss = model(
        exp=noisy_exp,
        img_features=embeddings,
        coords=coords,
        labels=labels,
        t_steps=t_steps,
    )
    return loss


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    interpolant = kwargs["interpolant"]
    n_steps = int(getattr(args, "n_sample_steps", 5))

    all_pred = []
    all_gt = []
    device = accelerator.device
    for batch in eval_dataloader:
        embeddings = _to_3d(batch["embeddings"]).to(device)
        coords = _to_3d(batch["coords"]).to(device)
        labels = _to_3d(batch["labels"]).to(device)

        # Iterative denoising starting from the trained prior
        x_t = interpolant.sample_from_prior(labels.shape, device).to(labels.dtype)
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t = ts[i].expand(embeddings.shape[0])
            d_t = (ts[i + 1] - ts[i]).expand(embeddings.shape[0])
            pred = (
                model.module.inference(x_t, embeddings, coords, t)
                if hasattr(model, "module")
                else model.inference(x_t, embeddings, coords, t)
            )
            x_t = interpolant.step(pred, x_t, t, d_t)

        pad_mask = embeddings.sum(-1) == 0
        # squeeze batch dim and select valid spots
        pred_flat = x_t[~pad_mask].cpu().numpy()
        gt_flat = labels[~pad_mask].cpu().numpy()
        all_pred.append(pred_flat)
        all_gt.append(gt_flat)

    return np.concatenate(all_pred, axis=0), np.concatenate(all_gt, axis=0)
