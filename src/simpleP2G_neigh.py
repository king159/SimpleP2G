"""SimpleP2G-cross: target tokens and spatially positioned neighbor tokens."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.simpleP2G import SimpleP2GDecoderLayer, SimpleP2GRMSNorm


class SimpleP2GNeigh(nn.Module):
    """The target-plus-neighbors model used in the main comparison."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder_depth = config.decoder_depth
        self.num_genes = config.num_genes
        self.hidden_size = config.hidden_size
        self.cond_size = config.cond_size
        self.num_cond_tokens = getattr(config, "num_cond_tokens", 4)
        self.num_neighbor_tokens = int(getattr(config, "num_neighbor_tokens", 24))
        if str(getattr(config, "neighbor_position_mode", "relative")) != "relative":
            raise ValueError("SimpleP2G-cross requires relative neighbor positions")
        for key in ("neighbor_cond_size", "global_cond_size"):
            if int(getattr(config, key, self.cond_size)) != self.cond_size:
                raise ValueError("All image embeddings must use the same encoder dimension")
        if (
            not getattr(config, "use_target_context", True)
            or not getattr(config, "use_neighbor_context", True)
            or getattr(config, "use_slide_context", False)
            or getattr(config, "use_local_summary", False)
            or getattr(config, "context_source_balance", 0.0) != 0
        ):
            raise ValueError(
                "This model requires target + neighbor context; use the separate experiment entry point for other variants"
            )
        self.gene_queries = nn.Embedding(self.num_genes, self.hidden_size)
        self.label_embed = nn.Sequential(
            nn.Linear(self.cond_size, self.cond_size),
            nn.SiLU(),
            nn.Linear(self.cond_size, self.hidden_size * self.num_cond_tokens),
        )
        self.neighbor_proj = nn.Sequential(
            nn.Linear(self.cond_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.neighbor_pos_proj = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.decoder_blocks = nn.ModuleList([SimpleP2GDecoderLayer(config) for _ in range(self.decoder_depth)])
        self.final_norm = SimpleP2GRMSNorm(self.hidden_size)
        self.mean_head = nn.Linear(self.hidden_size, 1)
        if config.loss_name in ("gaussian_nll", "beta_nll"):
            self.log_var_head = nn.Linear(self.hidden_size, 1)
        self.gradient_checkpointing = False
        self.emb_dropout = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, c, neighbor_emb=None, neighbor_mask=None, neighbor_pos=None, global_emb=None):
        """Predict from target (B,D), neighbors (B,K,D) and relative positions.

        global_emb is accepted for the shared slide-dataset interface; the
        main-comparison model does not consume a slide token.
        """
        if neighbor_emb is None or neighbor_pos is None:
            raise ValueError("neighbor_emb and neighbor_pos are required")
        batch_size = c.shape[0]
        output = self.emb_dropout(self.gene_queries.weight.unsqueeze(0).expand(batch_size, -1, -1))
        target = self.label_embed(c).view(batch_size, self.num_cond_tokens, self.hidden_size)
        neighbors = self.neighbor_proj(neighbor_emb)
        neighbors = neighbors + self.neighbor_pos_proj(neighbor_pos.to(dtype=neighbors.dtype))
        condition = torch.cat((target, neighbors), dim=1)
        if neighbor_mask is None:
            neighbor_mask = torch.ones(neighbor_emb.shape[:2], device=c.device, dtype=torch.bool)
        target_mask = torch.ones(batch_size, self.num_cond_tokens, device=c.device, dtype=torch.bool)
        mask = torch.cat((target_mask, neighbor_mask.bool()), dim=1).unsqueeze(1).unsqueeze(1)
        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                output = checkpoint(block, output, condition, None, mask, use_reentrant=False)
            else:
                output = block(output, encoder_hidden_states=condition, cross_attention_mask=mask)
        output = self.final_norm(output)
        mean = F.softplus(self.mean_head(output).squeeze(-1))
        log_var = (
            self.log_var_head(output).squeeze(-1)
            if self.config.loss_name in ("gaussian_nll", "beta_nll")
            else torch.zeros_like(mean)
        )
        return mean, log_var
