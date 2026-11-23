"""Frame averaging module — verbatim from the released STFlow stflow/model/fa.py."""

import torch
import torch.nn as nn
from einops import rearrange


class FrameAveraging(nn.Module):
    def __init__(self, dim=2, backward=False):
        super().__init__()
        self.dim = dim
        self.n_frames = 2**dim
        self.ops = self.create_ops(dim)  # [2^dim, dim]
        self.backward = backward

    def create_ops(self, dim):
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])
        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[dim_slice])
        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim=-1)
        operations = rearrange(operations, "... d -> (...) d")
        return operations

    def create_frame(self, X, mask=None):
        assert X.shape[-1] == self.dim
        # eigh is not implemented for half/bfloat16 — disable autocast & use fp32.
        orig_dtype = X.dtype
        with torch.amp.autocast(device_type=X.device.type, enabled=False):
            X = X.float()
            if mask is None:
                mask = torch.ones(*X.shape[:-1], device=X.device).bool()
            mask = mask.unsqueeze(-1)
            center = (X * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            X = X - center.unsqueeze(1) * mask
            X_ = X.masked_fill(~mask, 0.0)
            C = torch.bmm(X_.transpose(1, 2), X_)
            if not self.backward:
                C = C.detach()
            _, eigenvectors = torch.linalg.eigh(C, UPLO="U")
            F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(X.device) * eigenvectors.unsqueeze(1)
            h = torch.einsum("boij,bpj->bopi", F_ops.transpose(2, 3), X)
            h = h.view(X.size(0) * self.n_frames, X.size(1), self.dim)
        return h.to(orig_dtype), F_ops.detach().to(orig_dtype), center.to(orig_dtype)
