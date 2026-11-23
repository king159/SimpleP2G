"""Spatial Transformer (STFlow) — adapted from the released STFlow stflow/model/transformer.py.

* ``to_dense_batch`` (originally from torch_geometric) reimplemented locally.
* Coord dim defaults to 2 (we feed 2-D pixel coords).
* ``GeneUpdate`` no longer takes a ``non_negative`` arg (the original repo
  declared it but the class did not consume it).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange
from timm.models.vision_transformer import Mlp, SwiGLUPacked

from .fa import FrameAveraging


def _to_dense_batch(x: torch.Tensor, batch: torch.Tensor, max_num_nodes: int, fill_value: float = 0.0):
    """Pad-out flat features keyed by batch index to (B, N_max, D).

    Replacement for ``torch_geometric.utils.to_dense_batch``.
    """
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    D = x.shape[-1]
    out = x.new_full((B, max_num_nodes, D), fill_value)
    counts = torch.zeros(B, dtype=torch.long, device=x.device)
    # Position of each node within its slide
    for b in range(B):
        idx = (batch == b).nonzero(as_tuple=True)[0]
        n = idx.numel()
        out[b, :n] = x[idx]
        counts[b] = n
    mask = torch.arange(max_num_nodes, device=x.device).unsqueeze(0) < counts.unsqueeze(1)
    return out, mask


def _get_activation(activation="gelu"):
    return {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}[activation]


class GeneUpdate(nn.Module):
    def __init__(self, d_model, n_genes, proj_drop=0.0):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )

    def forward(self, features):
        return self.output(features)


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(self, d_model, d_edge_model, n_genes, n_heads=1, proj_drop=0.0, attn_drop=0.0, activation="gelu"):
        super().__init__(dim=2)
        self.d_head = d_model // n_heads
        self.d_edge_head = d_edge_model // n_heads
        self.n_heads = n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.d_head * 2 + self.d_edge_head + n_genes,
                hidden_features=d_model,
                out_features=1,
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim + 1,
                hidden_features=d_edge_model,
                out_features=d_edge_model,
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model + d_edge_model,
                hidden_features=d_model,
                out_features=d_model,
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.d_head * 2 + self.d_edge_head + n_genes,
                hidden_features=d_model,
                out_features=1,
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
            self.edge_trans = Mlp(
                in_features=self.dim + 1,
                hidden_features=d_edge_model,
                out_features=d_edge_model,
                act_layer=_get_activation(activation),
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
            self.W_output = Mlp(
                in_features=d_model + d_edge_model,
                hidden_features=d_model,
                out_features=d_model,
                act_layer=_get_activation(activation),
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None):
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = (rearrange(t, "n (h d) -> n h d", h=n_heads) for t in (q_s, k_s, v_s))

        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)

        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)

        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)

        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))

        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)
        scalar_context = einsum(attn_map, v_s_neighs, "n h m, n m h d -> n h d").view(n_tokens, -1)
        edge_context = einsum(attn_map, frame_feats, "n h m, n m h d -> n h d").view(n_tokens, -1)
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_edge_model,
        n_genes,
        n_heads=1,
        activation="gelu",
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model,
            d_edge_model=d_edge_model,
            n_genes=n_genes,
            n_heads=n_heads,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            activation=activation,
        )
        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model,
                hidden_features=int(d_model * mlp_ratio),
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
        else:
            self.mlp = Mlp(
                in_features=d_model,
                hidden_features=int(d_model * mlp_ratio),
                act_layer=_get_activation(activation),
                drop=proj_drop,
                norm_layer=nn.LayerNorm,
            )
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices):
        context_token_embs = self.attn(gene_exp, token_embs, coords, neighbor_indices)
        token_embs = token_embs + context_token_embs
        token_embs = token_embs + self.mlp(token_embs)
        gene_exp = self.gene_updater(token_embs)
        return gene_exp, token_embs


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_input: int,
        d_model: int = 128,
        d_edge_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.2,
        n_neighbors: int = 8,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        # NB: feature input dim is consumed by the upstream Denoiser via image_transform;
        # blocks operate in d_model space.
        self.blks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    d_edge_model=d_edge_model,
                    n_genes=n_genes,
                    n_heads=n_heads,
                    activation=activation,
                    attn_drop=attn_dropout,
                    proj_drop=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def _build_graph(self, coords, batch_idx, n_neighbors, exclude_self=True):
        N = coords.shape[0]
        exclude_self_mask = torch.eye(N, dtype=torch.bool, device=coords.device)
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)
        rel_pos = rearrange(coords, "n d -> n 1 d") - rearrange(coords, "n d -> 1 n d")
        rel_dist = rel_pos.norm(dim=-1).detach()
        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)
        _, nearest_indices = rel_dist.topk(n_neighbors, dim=-1, largest=False)
        return nearest_indices

    def forward(self, gene_exp, features, coords):
        # gene_exp: [B, N_cells, N_genes]; features: [B, N_cells, d_model]; coords: [B, N_cells, 2]
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device
        pad_mask = features.sum(dim=-1) == 0  # [B, N_cells]
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]

        features = features[~pad_mask]
        coords = coords[~pad_mask]
        gene_exp = gene_exp[~pad_mask]

        nearest_indices = self._build_graph(
            coords, batch_idx, min(self.n_neighbors, features.shape[0] - 1), exclude_self=True
        )

        all_gene_exp = []
        for blk in self.blks:
            gene_exp, features = blk(gene_exp, features, coords, nearest_indices)
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)

        gene_exp_dense, _ = _to_dense_batch(gene_exp, batch_idx, max_num_nodes=N_cells, fill_value=0.0)
        return gene_exp_dense
