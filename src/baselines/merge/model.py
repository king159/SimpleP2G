"""MERGE GAT model + hierarchical graph construction."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GATConv, LayerNorm
from torch_geometric.utils import dropout_edge, from_scipy_sparse_matrix


class MERGEModel(nn.Module):
    """Linear projection (feature_dim -> 256) + 4-layer GATNet head.

    The original repository's GATNet is reproduced verbatim; only the input
    projection differs (they used a SimCLR-finetuned ResNet18 to produce 256-d
    features; we project precomputed embeddings instead).
    """

    def __init__(
        self,
        feature_dim: int,
        num_genes: int,
        num_heads: int = 8,
        drop_edge: float = 0.2,
    ):
        super().__init__()
        # The released MERGE encoder already emits 256-d features. Foundation
        # encoders require the smallest possible adapter to preserve the
        # original GAT input contract.
        self.input_proj = (
            nn.Identity()
            if feature_dim == 256
            else nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
            )
        )

        dim1, dim2, dim3 = 448, 384, 256
        self.drop_edge = drop_edge
        self.nn1 = GATConv(256, dim1, num_heads)
        self.layer_norm1 = LayerNorm(dim1 * num_heads)
        self.nn2 = GATConv(dim1 * num_heads, dim2, num_heads)
        self.layer_norm2 = LayerNorm(dim2 * num_heads)
        self.nn3 = GATConv(dim2 * num_heads, dim3, num_heads)
        self.layer_norm3 = LayerNorm(dim3 * num_heads)
        self.nn4 = GATConv(dim3 * num_heads, num_genes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # x: (N, feature_dim); edge_index: (2, E)
        x = self.input_proj(x)
        edge_index, _ = dropout_edge(edge_index, p=self.drop_edge, training=self.training)

        x = F.relu(self.nn1(x, edge_index))
        x = self.layer_norm1(x)
        x = F.relu(self.nn2(x, edge_index))
        x = self.layer_norm2(x)
        x = F.relu(self.nn3(x, edge_index))
        x = self.layer_norm3(x)
        x = self.nn4(x, edge_index)
        return x


# ──────────────────────────────────────────────────────────────────────
# Hierarchical graph construction (port of the released MERGE utils/graph.py)
# ──────────────────────────────────────────────────────────────────────


def _update_adj(
    adj: np.ndarray,
    cluster_labels: np.ndarray,
    patch_embeddings: np.ndarray,
    old_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_labels = cluster_labels.astype(np.int64).copy()
    unique_cluster_labels = np.unique(cluster_labels)
    centroid_spots: list[int] = []

    for cluster_label in unique_cluster_labels:
        cluster_spots = np.where(cluster_labels == cluster_label)[0]
        if cluster_spots.size == 0:
            continue
        cluster_centroid = patch_embeddings[cluster_spots].mean(axis=0)
        nearest_spot_idx = int(np.argmin(np.linalg.norm(patch_embeddings[cluster_spots] - cluster_centroid, axis=1)))
        nearest_spot = int(cluster_spots[nearest_spot_idx])
        for j in cluster_spots:
            if j != nearest_spot:
                adj[j, nearest_spot] = 1
                adj[nearest_spot, j] = 1
        centroid_spots.append(nearest_spot)
        cluster_labels[nearest_spot] *= -1
        if cluster_labels[nearest_spot] == 0:
            cluster_labels[nearest_spot] = -(len(unique_cluster_labels))

    if old_labels is not None:
        for j, old_label in enumerate(old_labels):
            if old_label < 0:
                centroid_spots.append(int(j))

    centroid_spots = list(set(centroid_spots))
    for j in range(len(centroid_spots)):
        for k in range(j + 1, len(centroid_spots)):
            adj[centroid_spots[j], centroid_spots[k]] = 1
            adj[centroid_spots[k], centroid_spots[j]] = 1
    return adj, cluster_labels


def build_hierarchical_edges(
    coords: np.ndarray,
    embeddings: np.ndarray,
    n_neighbors: int = 8,
    n_spatial_clusters: int = 5,
    n_feature_clusters: int = 5,
    seed: int = 0,
) -> torch.Tensor:
    """Build a hierarchical adjacency for MERGE.

    Steps (from MERGE paper, Sec. graph construction):
      1. 8-NN spatial graph (symmetric, with self-loops).
      2. Spatial KMeans -> add edges from each spot to its cluster's centroid
         spot, and connect all centroids to each other.
      3. Feature KMeans -> repeat the centroid-edge step on top of (2).

    Returns a ``(2, E)`` edge index tensor.
    """
    n = coords.shape[0]
    n_spatial_clusters = min(n_spatial_clusters, max(n - 1, 1))
    n_feature_clusters = min(n_feature_clusters, max(n - 1, 1))
    n_neighbors = min(n_neighbors, max(n - 1, 1))

    # 1) k-NN spatial graph
    knn = kneighbors_graph(coords, mode="connectivity", n_neighbors=n_neighbors).toarray()
    adj = ((knn + knn.T) > 0).astype(np.uint8)
    np.fill_diagonal(adj, 1)

    # 2) Spatial KMeans
    km_spatial = KMeans(n_clusters=n_spatial_clusters, max_iter=300, n_init=5, random_state=seed)
    spatial_labels = km_spatial.fit_predict(coords)
    adj, spatial_labels = _update_adj(adj, spatial_labels, embeddings, None)

    # 3) Feature KMeans
    km_feature = KMeans(n_clusters=n_feature_clusters, max_iter=300, n_init=5, random_state=seed)
    feature_labels = km_feature.fit_predict(embeddings)
    adj, _ = _update_adj(adj, feature_labels, embeddings, spatial_labels)

    edge_index, _ = from_scipy_sparse_matrix(sp.coo_matrix(adj))
    return edge_index.long()


def build_merge_from_omegaconf(args):
    return MERGEModel(
        feature_dim=int(args.cond_size),
        num_genes=int(args.input_gene_size),
        num_heads=int(getattr(args, "gat_num_heads", 8)),
        drop_edge=float(getattr(args, "drop_edge", 0.2)),
    )
