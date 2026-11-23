"""Train / eval API for the MERGE baseline.

We pre-build the hierarchical edge index per slide once (cached in the dataset
batch dict) and run the GAT slide-by-slide. Loss is MSE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.merge.model import build_hierarchical_edges


_EDGE_CACHE: dict[tuple, torch.Tensor] = {}
_DEVICE_EDGE_CACHE: dict[tuple[tuple, str], torch.Tensor] = {}


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    """Drop a leading B=1 dim if present (DataLoader adds one with batch_size=1)."""
    if t.dim() == 3:
        return t.squeeze(0)
    return t


def _build_edge_index(batch, args, device):
    """Build edges once per fixed full slide and reuse them across epochs.

    Keep one device-resident copy as well as the CPU copy.  Calling
    ``cpu_edge.to(device)`` for every step creates a distinct CUDA tensor; the
    PyG GAT backward path can retain those allocations long enough for a
    full-slide run to grow monotonically across epochs.  Reusing the immutable
    edge tensor removes that growth without changing graph construction.
    """
    if "edge_index" in batch:
        ei = batch["edge_index"]
        ei = _ensure_2d(ei) if isinstance(ei, torch.Tensor) else ei
        return ei.to(device)

    slide_id = batch.get("slide_id", "")
    if isinstance(slide_id, (list, tuple)):
        slide_id = slide_id[0]
    cacheable = not int(getattr(args, "max_spots", 0))
    cache_key = (
        str(slide_id),
        int(_ensure_2d(batch["embeddings"]).shape[0]),
        int(getattr(args, "n_neighbors", 8)),
        int(getattr(args, "n_spatial_clusters", 5)),
        int(getattr(args, "n_feature_clusters", 5)),
        int(getattr(args, "seed", 0)),
    )
    if cacheable and cache_key in _EDGE_CACHE:
        device_key = (cache_key, str(device))
        if device_key not in _DEVICE_EDGE_CACHE:
            _DEVICE_EDGE_CACHE[device_key] = _EDGE_CACHE[cache_key].to(device)
        return _DEVICE_EDGE_CACHE[device_key]

    coords = _ensure_2d(batch["coords"]).cpu().numpy()
    embeddings = _ensure_2d(batch["embeddings"]).cpu().numpy()
    edge_index = build_hierarchical_edges(
        coords=coords,
        embeddings=embeddings,
        n_neighbors=int(getattr(args, "n_neighbors", 8)),
        n_spatial_clusters=int(getattr(args, "n_spatial_clusters", 5)),
        n_feature_clusters=int(getattr(args, "n_feature_clusters", 5)),
        seed=int(getattr(args, "seed", 0)),
    )
    if cacheable:
        cpu_edge_index = edge_index.cpu()
        _EDGE_CACHE[cache_key] = cpu_edge_index
        device_key = (cache_key, str(device))
        _DEVICE_EDGE_CACHE[device_key] = cpu_edge_index.to(device)
        return _DEVICE_EDGE_CACHE[device_key]
    return edge_index.to(device)


def train_one_step(model, batch, args, **kwargs):
    embeddings = _ensure_2d(batch["embeddings"])  # (N, D)
    labels = _ensure_2d(batch["labels"])  # (N, G)
    device = embeddings.device
    edge_index = _build_edge_index(batch, args, device)

    pred = model(embeddings, edge_index)
    return F.mse_loss(pred, labels)


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    device = accelerator.device
    all_pred, all_gt = [], []
    for batch in eval_dataloader:
        embeddings = _ensure_2d(batch["embeddings"]).to(device)
        labels = _ensure_2d(batch["labels"]).to(device)
        edge_index = _build_edge_index(batch, args, device)
        pred = model.module(embeddings, edge_index) if hasattr(model, "module") else model(embeddings, edge_index)
        all_pred.append(pred.cpu().numpy())
        all_gt.append(labels.cpu().numpy())
    return np.concatenate(all_pred, axis=0), np.concatenate(all_gt, axis=0)
