"""Official STRankList-v2 objective, with batched gathers.

Source: naivete5656/STRank, commit e1917b4b311a72d9150c03bc4c546d1a6550888e,
strank/loss_functions.py: group_perm, calc_raw_lpi_norm, STRankLoss.
The official training step sums rank_binom_loss and pi_prior. The grouping,
CPU randperm draw order, count normalization, reductions and prior are retained.
Only redundant device transfers and gathers are batched for efficiency.
"""

import torch


def official_strank_v2_terms(preds, counts, sample_ids, n_pair=256):
    if preds.ndim != 2 or counts.shape != preds.shape:
        raise ValueError("STRank expects aligned spot-by-gene scores and raw counts")
    if sample_ids.ndim != 1 or len(sample_ids) != len(preds) or n_pair < 1:
        raise ValueError("Supply a training-slide ID for each spot and n_pair >= 1")
    groups = sample_ids.detach().cpu()
    indices = [torch.where(groups == group)[0] for group in torch.unique(groups)]
    permutations = torch.empty((n_pair, len(groups)), dtype=torch.long)
    # Same CPU RNG calls, in the same order, as repeated official group_perm.
    for repeat in range(n_pair):
        for index in indices:
            permutations[repeat, index] = index[torch.randperm(len(index))]
    index = permutations.T.to(preds.device)
    stack_count = counts[index]
    stack_pred = preds[index]
    raw_lpi = stack_pred + (stack_count.float().mean(dim=-1, keepdim=True) + 1e-6).log()
    lpi = torch.nn.functional.log_softmax(raw_lpi, dim=-2)
    return {"rank_binom_loss": (-lpi * stack_count).mean(),
            "pi_prior": -(lpi.exp() * lpi).mean()}
