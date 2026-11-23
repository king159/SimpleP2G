"""Train and evaluate the official STRank objective under the shared benchmark."""

import numpy as np
import torch

from src.baselines.strank.model import get_strank_loss


def train_one_step(model, batch, args, **kwargs):
    pred = model(batch["img_feat"])
    return get_strank_loss(args, pred, batch["count"], sample_ids=batch["sample_id"])


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    predictions, targets = [], []
    for batch in eval_dataloader:
        # Raw counts, slide IDs and expression labels never enter the predictor.
        predictions.append(model(batch["img_feat"]).detach().cpu())
        targets.append(batch["labels"].cpu().numpy())
    return torch.cat(predictions).numpy(), np.concatenate(targets)
