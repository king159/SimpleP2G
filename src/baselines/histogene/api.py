"""HisToGene training / evaluation helpers.

Both ``histogene_enhanced`` and ``histogene_original`` use slide-level batching:
each item from the ``HisToGeneSlideDataset`` is a *full slide* with shapes
``(1, N_spots, ...)``.  The ``DataLoader`` uses ``batch_size=1``.
"""

from __future__ import annotations

import torch


def _unwrap(model):
    """Unwrap torch.compile + DDP/FSDP wrappers to access custom methods."""
    m = model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if hasattr(m, "module"):
        m = m.module
    return m


def train_one_step(model, batch: dict, args, **kwargs) -> torch.Tensor:
    """Single training step – handles both enhanced and original modes.

    ``batch`` comes from ``HisToGeneSlideDataset.__getitem__`` and has
    a leading batch dimension of 1 added by the DataLoader.
    """
    labels = batch["labels"]  # (1, N, G)
    pos_bins = batch["pos_bins"]  # (1, N, 2)

    model_name = str(getattr(args, "model", "")).lower()

    if model_name in {"histogene", "histogene_enhanced"}:
        features = batch["embeddings"]  # (1, N, D)
    else:  # histogene_original
        features = batch["patches"]  # (1, N, patch_dim)

    raw_model = _unwrap(model)
    loss = raw_model.compute_loss(features, pos_bins, labels)
    return loss


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    """Evaluate across all held-out slides.

    Each slide is processed as one batch.  Returns concatenated per-spot
    predictions and ground truth.
    """
    model_name = str(getattr(args, "model", "")).lower()
    all_pred = []
    all_true = []

    for batch in eval_dataloader:
        # Move to device
        batch = {k: v.to(accelerator.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        labels = batch["labels"]  # (1, N, G)
        pos_bins = batch["pos_bins"]  # (1, N, 2)

        if model_name in {"histogene", "histogene_enhanced"}:
            features = batch["embeddings"]
        else:
            features = batch["patches"]

        raw_model = _unwrap(model)
        pred = raw_model(features, pos_bins)  # (1, N, G)

        # Flatten the slide dimension → (N, G)
        all_pred.append(pred.squeeze(0).detach().cpu())
        all_true.append(labels.squeeze(0).detach().cpu())

    pred = torch.cat(all_pred, dim=0).numpy()
    test_count_mtx_selected_genes = torch.cat(all_true, dim=0).numpy()
    return pred, test_count_mtx_selected_genes
