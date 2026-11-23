"""M2TGLGO HEST adapter.

Reference implementation: https://github.com/shihangjs/M2TGLGO

The official method uses multi-modal graph convolution over cell morphology,
patch morphology, image features, and a gene ontology/gene2vec graph. Our HEST
cache does not contain Hover-Net cell features or GO/gene2vec assets, and DGL is
not installed. This adapter ports the multi-modal graph fusion idea to
PyTorch-Geometric using available HEST features: spot embeddings, coordinate
morphology proxies, local density features, and a learnable gene-relation table.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, LayerNorm


class ModalityAttention(nn.Module):
    def __init__(self, dim: int, n_modalities: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_modalities)])
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        hs = [torch.tanh(p(x)) for p, x in zip(self.proj, xs)]
        stack = torch.stack(hs, dim=1)
        weights = torch.softmax(self.score(stack), dim=1)
        raw = torch.stack(xs, dim=1)
        return (weights * raw).sum(dim=1)


class GraphBlock(nn.Module):
    def __init__(self, dim: int, gnn_type: str = "gat", heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = float(dropout)
        gnn_type = gnn_type.lower()
        if gnn_type == "gcn":
            self.conv = GCNConv(dim, dim)
        elif gnn_type == "gat":
            out = max(dim // heads, 1)
            self.conv = GATConv(dim, out, heads=heads, dropout=dropout)
            self.out_proj = nn.Linear(out * heads, dim) if out * heads != dim else nn.Identity()
        else:
            raise ValueError(f"Unsupported m2tglgo gnn_type={gnn_type!r}")
        self.gnn_type = gnn_type
        self.norm = LayerNorm(dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index)
        if self.gnn_type == "gat":
            h = self.out_proj(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.norm(h + x)


class M2TGLGOAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_genes: int,
        hidden_dim: int = 512,
        gene_dim: int = 512,
        gnn_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.2,
        alpha: float = 0.75,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.image_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.coord_proj = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.neigh_proj = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.blocks = nn.ModuleList([GraphBlock(hidden_dim, gnn_type, heads, dropout) for _ in range(3)])
        self.inter_attn = ModalityAttention(hidden_dim, 3)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.gene_embeddings = nn.Parameter(torch.randn(num_genes, gene_dim) * 0.02)
        self.node_to_gene_dim = nn.Linear(hidden_dim, gene_dim)
        self.bias = nn.Parameter(torch.zeros(num_genes))

    @staticmethod
    def coord_features(coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, :1]
        y = coords[:, 1:2]
        return torch.cat(
            [
                x,
                y,
                x * y,
                x.square(),
                y.square(),
                torch.sin(torch.pi * x),
                torch.cos(torch.pi * x),
                torch.sin(torch.pi * y),
                torch.cos(torch.pi * y),
                torch.ones_like(x),
            ],
            dim=1,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        neighbor_mean: torch.Tensor,
    ):
        modalities = [
            self.image_proj(embeddings),
            self.coord_proj(self.coord_features(coords)),
            self.neigh_proj(neighbor_mean),
        ]
        intra = [block(x, spatial_edge_index) for block, x in zip(self.blocks, modalities)]
        inter = self.inter_attn(modalities)
        fused_modalities = [self.alpha * a + (1.0 - self.alpha) * inter for a in intra]
        node = self.fuse(self.inter_attn(fused_modalities))
        node_gene = self.node_to_gene_dim(node)
        gene = F.normalize(self.gene_embeddings, dim=-1)
        return node_gene @ gene.T + self.bias, node_gene


def build_m2tglgo_from_omegaconf(args):
    return M2TGLGOAdapter(
        feature_dim=int(args.cond_size),
        num_genes=int(args.input_gene_size),
        hidden_dim=int(getattr(args, "hidden_dim", 512)),
        gene_dim=int(getattr(args, "m2_gene_dim", 512)),
        gnn_type=str(getattr(args, "m2_gnn_type", "gat")),
        heads=int(getattr(args, "gat_heads", 4)),
        dropout=float(getattr(args, "dropout", 0.2)),
        alpha=float(getattr(args, "m2_alpha", 0.75)),
    )
