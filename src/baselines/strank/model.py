"""Official STRank linear head and STRankList-v2 training objective.

Pinned reference: naivete5656/STRank, e1917b4b311a72d9150c03bc4c546d1a6550888e.
Each prediction uses only the target image embedding. Raw counts and slide IDs
are supplied exclusively to the training objective.
"""

import torch
from torch import nn

from src.baselines.strank.official import official_strank_v2_terms


class STRankLinear(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(int(args.cond_size), int(args.input_gene_size))

    def forward(self, features):
        return self.linear(features)


def get_strank_loss(args, preds, counts, *, sample_ids):
    if str(args.loss_name) != "stranklist_v2":
        raise ValueError("STRank implements the official stranklist_v2 objective")
    scores = preds.float() if preds.dtype in (torch.float16, torch.bfloat16) else preds
    return sum(official_strank_v2_terms(scores, counts, sample_ids,
                                      n_pair=int(args.strank_n_pair)).values())
