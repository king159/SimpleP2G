"""Train / eval API for the M2TGLGO HEST adapter."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t.squeeze(0) if t.dim() == 3 else t


def _spatial_edges(coords: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    x_np = coords.detach().cpu().numpy()
    n = x_np.shape[0]
    if n <= 1:
        return torch.zeros((2, 1), dtype=torch.long, device=device)
    k = min(int(k), n - 1)
    adj = kneighbors_graph(x_np, n_neighbors=k, mode="connectivity", include_self=True)
    adj = ((adj + adj.T) > 0).astype(np.uint8)
    ei, _ = from_scipy_sparse_matrix(adj.tocoo())
    return ei.long().to(device)


def _neighbor_mean(embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    out = torch.zeros_like(embeddings)
    deg = torch.zeros(embeddings.size(0), 1, device=embeddings.device, dtype=embeddings.dtype)
    out.index_add_(0, dst, embeddings[src])
    deg.index_add_(0, dst, torch.ones_like(deg[src]))
    return out / deg.clamp_min(1.0)


def _batch_inputs(batch, args, device):
    embeddings = _ensure_2d(batch["embeddings"]).to(device)
    coords = _ensure_2d(batch["coords"]).to(device)
    edge_index = _spatial_edges(coords, int(getattr(args, "n_neighbors", 8)), device)
    neighbor_mean = _neighbor_mean(embeddings, edge_index)
    return embeddings, coords, edge_index, neighbor_mean


def _adjacency_loss(node: torch.Tensor, edge_index: torch.Tensor, n_negative: int = 4096) -> torch.Tensor:
    if edge_index.numel() == 0:
        return node.sum() * 0.0
    src, dst = edge_index
    pos = (F.normalize(node[src], dim=-1) * F.normalize(node[dst], dim=-1)).sum(dim=-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    n = node.size(0)
    m = min(int(n_negative), max(src.numel(), 1))
    neg_src = torch.randint(0, n, (m,), device=node.device)
    neg_dst = torch.randint(0, n, (m,), device=node.device)
    neg = (F.normalize(node[neg_src], dim=-1) * F.normalize(node[neg_dst], dim=-1)).sum(dim=-1)
    neg_loss = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    return 0.5 * (pos_loss + neg_loss)


def train_one_step(model, batch, args, **kwargs):
    device = _ensure_2d(batch["embeddings"]).device
    labels = _ensure_2d(batch["labels"]).to(device)
    embeddings, coords, edge_index, neighbor_mean = _batch_inputs(batch, args, device)
    pred, node = model(embeddings, coords, edge_index, neighbor_mean)
    expr_loss = F.mse_loss(pred, labels)
    graph_loss = _adjacency_loss(node, edge_index) * float(getattr(args, "m2_graph_loss_weight", 0.01))
    return expr_loss + graph_loss


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    device = accelerator.device
    raw_model = model.module if hasattr(model, "module") else model
    all_pred, all_gt = [], []
    for batch in eval_dataloader:
        labels = _ensure_2d(batch["labels"]).to(device)
        embeddings, coords, edge_index, neighbor_mean = _batch_inputs(batch, args, device)
        pred, _ = raw_model(embeddings, coords, edge_index, neighbor_mean)
        all_pred.append(pred.cpu().numpy())
        all_gt.append(labels.cpu().numpy())
    return np.concatenate(all_pred, axis=0), np.concatenate(all_gt, axis=0)
