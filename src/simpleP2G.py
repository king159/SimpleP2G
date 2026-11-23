from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleP2GRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SimpleP2GMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = getattr(
            config,
            "intermediate_size",
            int(config.hidden_size * getattr(config, "mlp_ratio", 4.0)),
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(down_proj)


class SimpleP2GAttention(nn.Module):
    """
    multi-head self-attention module using F.scaled_dot_product_attention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Prepare attention mask for F.scaled_dot_product_attention
        # F.sdpa expects mask to be True for positions that should attend
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # (B, N)
                # Convert to (B, 1, 1, N) and invert (0 -> False, 1 -> True)
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:  # (B, 1, N)
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:  # (B, 1, 1, N) or (B, H, N, N)
                attention_mask = attention_mask.bool()

        # Use F.scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class SimpleP2GCrossAttention(nn.Module):
    """
    Multi-head cross-attention module.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Query
        encoder_hidden_states: torch.Tensor,  # Key, Value
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        encoder_seq_len = encoder_hidden_states.size(1)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, encoder_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, encoder_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if attention_mask is not None and not (
            attention_mask.is_floating_point() or attention_mask.dtype == torch.bool
        ):
            attention_mask = attention_mask.bool()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class SimpleP2GDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        eps = getattr(config, "rms_norm_eps", 1e-6)

        # Layers
        self.self_attn = SimpleP2GAttention(config=config)
        self.cross_attn = SimpleP2GCrossAttention(config=config)
        self.mlp = SimpleP2GMLP(config)

        # Pre-Norms (Normalization happens BEFORE the sub-layer)
        self.norm1 = SimpleP2GRMSNorm(self.hidden_size, eps=eps)
        self.norm2 = SimpleP2GRMSNorm(self.hidden_size, eps=eps)
        self.norm3 = SimpleP2GRMSNorm(self.hidden_size, eps=eps)
        self.resid_dropout = nn.Dropout(getattr(config, "dropout", 0.0))

    def forward(self, x, encoder_hidden_states, attention_mask=None, cross_attention_mask=None):
        # 1. Self-Attention (Genes interacting with each other)
        x = x + self.resid_dropout(self.self_attn(self.norm1(x), attention_mask=attention_mask))

        # 2. Cross-Attention (Genes looking at the image context)
        x = x + self.resid_dropout(
            self.cross_attn(self.norm2(x), encoder_hidden_states, attention_mask=cross_attention_mask)
        )

        # 3. Feed-Forward
        x = x + self.mlp(self.norm3(x))

        return x


class SimpleP2G(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config
        self.decoder_depth = config.decoder_depth
        self.num_genes = config.num_genes
        self.hidden_size = config.hidden_size
        self.gene_queries = nn.Embedding(self.num_genes, self.hidden_size)
        self.cond_size = config.cond_size
        self.num_cond_tokens = getattr(config, "num_cond_tokens", 1)
        self.decoder_blocks = nn.ModuleList([SimpleP2GDecoderLayer(config) for _ in range(self.decoder_depth)])
        self.final_norm = SimpleP2GRMSNorm(self.hidden_size)
        self.mean_head = nn.Linear(self.hidden_size, 1)
        if config.loss_name in ["gaussian_nll", "beta_nll"]:
            self.log_var_head = nn.Linear(self.hidden_size, 1)
        # if config.loss_name == "zinb":
        #     self.pi_head = nn.Linear(self.hidden_size, 1)
        self.gradient_checkpointing = False
        self.emb_dropout = nn.Dropout(getattr(config, "dropout", 0.0))

        # ── Additional regularization ──
        self.gene_drop_rate = float(getattr(config, "gene_drop_rate", 0.0))
        self.emb_noise_std = float(getattr(config, "emb_noise_std", 0.0))
        self.stochastic_depth_rate = float(getattr(config, "stochastic_depth_rate", 0.0))

        self.label_embed = nn.Sequential(
            nn.Linear(self.cond_size, self.cond_size, bias=True),
            nn.SiLU(),
            nn.Linear(self.cond_size, self.hidden_size * self.num_cond_tokens, bias=True),
        )

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass."""
        output = self.gene_queries.weight.unsqueeze(0).expand(c.size(0), -1, -1)  # (B, num_genes, hidden_size)
        output = self.emb_dropout(output)

        # Gene dropout: randomly zero out gene query positions during training
        if self.training and self.gene_drop_rate > 0:
            gene_mask = torch.rand(c.size(0), self.num_genes, 1, device=c.device) > self.gene_drop_rate
            output = output * gene_mask  # zero out dropped genes

        condition = self.label_embed(c).view(c.size(0), self.num_cond_tokens, self.hidden_size)  # (B, T, hidden_size)

        # Embedding noise: add Gaussian noise to conditioning tokens during training
        if self.training and self.emb_noise_std > 0:
            condition = condition + torch.randn_like(condition) * self.emb_noise_std
        for i, block in enumerate(self.decoder_blocks):
            # Stochastic depth: randomly skip layers during training
            if self.training and self.stochastic_depth_rate > 0:
                # Linear scaling: deeper layers have higher drop probability
                drop_prob = self.stochastic_depth_rate * i / max(len(self.decoder_blocks) - 1, 1)
                if torch.rand(1).item() < drop_prob:
                    continue

            if self.gradient_checkpointing and self.training:
                output = torch.utils.checkpoint.checkpoint(
                    block,
                    output,
                    encoder_hidden_states=condition,
                    use_reentrant=False,
                )
            else:
                output = block(
                    output,
                    encoder_hidden_states=condition,
                )
        output = self.final_norm(output)
        raw_mean = self.mean_head(output).squeeze(-1)
        mean = F.softplus(raw_mean)  # Ensures predicted G >= 0
        if self.config.loss_name in ["gaussian_nll", "beta_nll"]:
            log_var = self.log_var_head(output).squeeze(-1)
        # if self.config.loss_name == "zinb":
        #     pi_logits = self.pi_head(output).squeeze(-1)
        #     return mean, log_var, pi_logits
        elif self.config.loss_name in ["mse", "l1"]:
            log_var = torch.zeros_like(mean)
        return mean, log_var


def get_loss_fn(loss_type: str) -> nn.Module:
    if loss_type == "gaussian_nll":
        return GaussianNLLLoss()
    elif loss_type == "beta_nll":
        return BetaNLLLoss(beta=1.0)
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


class GaussianNLLLoss(nn.Module):
    def __init__(self, logvar_clipping=(-10, 10)):
        super().__init__()
        self.min_logvar, self.max_logvar = logvar_clipping

    def forward(self, pred_mean, pred_logvar, target):
        # 1. Clamp logvar to prevent NaN gradients or Inf precision
        pred_logvar = torch.clamp(pred_logvar, self.min_logvar, self.max_logvar)

        # 2. Precision = exp(-logvar) is 1/sigma^2
        precision = torch.exp(-pred_logvar)

        # 3. Standard NLL formula components
        # (target - mean)^2 / variance + log(variance)
        diff = (target - pred_mean) ** 2
        loss = 0.5 * (precision * diff + pred_logvar)

        return loss.mean()


class BetaNLLLoss(nn.Module):
    """
    Beta-Negative Log-Likelihood Loss.
    Reference: Seitzer et al., "Properties and Benefits of Training with
    Beta-Negative Log-Likelihood", ICLR 2022.
    """

    def __init__(self, beta=1.0, logvar_clipping=(-10, 10)):
        super().__init__()
        self.beta = beta
        self.min_logvar, self.max_logvar = logvar_clipping

    def forward(self, pred_mean, pred_logvar, target):
        """
        pred_mean: (B, num_genes)
        pred_logvar: (B, num_genes)
        target: (B, num_genes)
        """
        # 1. Clamp for stability
        pred_logvar = torch.clamp(pred_logvar, self.min_logvar, self.max_logvar)

        # 2. Calculate variance (sigma^2)
        variance = torch.exp(pred_logvar)

        # 3. Standard Gaussian NLL components
        # (y - mu)^2 / sigma^2 + log(sigma^2)
        squared_error = (target - pred_mean) ** 2
        nll = (squared_error / variance) + pred_logvar

        # 4. Apply Beta Weighting
        # The 'stop_gradient' (detach) on the multiplier is optional
        # but often used to stabilize the variance head.
        # The paper version uses the variance gradient as well.
        weight = variance.detach() ** self.beta

        loss = 0.5 * weight * nll

        return loss.mean()
