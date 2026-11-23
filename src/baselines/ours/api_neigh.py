"""
Train / eval API for simpleP2G-neigh.

The dataset returns dicts instead of tuples, so we unpack local embeddings,
their masks and relative positions, and the optional slide token alongside
the standard data (gene expression) and single-patch embedding.
"""

import numpy as np
import torch

from src.simpleP2G import get_loss_fn


def _unpack_batch(batch):
    """Return spot-level tensors from either spot batches or one-slide batches."""
    x = batch["labels"]
    y = batch["embeddings"]
    nb = batch["neighbor_emb"]
    nb_mask = batch["neighbor_mask"]
    nb_pos = batch.get("neighbor_pos")
    g = batch.get("global_emb")

    if y.dim() == 3:
        flat_spots = y.shape[0] * y.shape[1]
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        # Use an explicit leading size: PyTorch cannot infer ``-1`` when a
        # disabled context source contributes a zero-sized token dimension.
        nb = nb.reshape(flat_spots, nb.shape[-2], nb.shape[-1])
        nb_mask = nb_mask.reshape(flat_spots, nb_mask.shape[-1])
        if nb_pos is not None:
            nb_pos = nb_pos.reshape(
                flat_spots,
                nb_pos.shape[-2],
                nb_pos.shape[-1],
            )
        if g is not None:
            g = g.reshape(flat_spots, g.shape[-1])
    return x, y, nb, nb_mask, nb_pos, g


def train_one_step(model, batch, args, **kwargs):
    """One training step for simpleP2G-neigh."""
    x, y, nb, nb_mask, nb_pos, g = _unpack_batch(batch)

    mean_preds, log_var_pred = model(
        y,
        neighbor_emb=nb,
        neighbor_mask=nb_mask,
        neighbor_pos=nb_pos,
        global_emb=g,
    )

    if args.loss_name in ["gaussian_nll", "beta_nll"]:
        loss = get_loss_fn(args.loss_name)(mean_preds, log_var_pred, x)
    elif args.loss_name in ["mse", "l1"]:
        loss = get_loss_fn(args.loss_name)(mean_preds, x)
    return loss


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    """Evaluation step for simpleP2G-neigh."""
    all_samples = []
    test_count_mtx_selected_genes = []
    for batch in eval_dataloader:
        x, y, nb, nb_mask, nb_pos, g = _unpack_batch(batch)

        samples = model(
            y,
            neighbor_emb=nb,
            neighbor_mask=nb_mask,
            neighbor_pos=nb_pos,
            global_emb=g,
        )
        if isinstance(samples, tuple):
            samples = samples[0]
        all_samples.append(samples.detach().cpu())
        test_count_mtx_selected_genes.append(x.cpu().numpy())

    pred = torch.cat(all_samples, dim=0).numpy()
    test_count_mtx_selected_genes = np.concatenate(test_count_mtx_selected_genes, axis=0)
    return pred, test_count_mtx_selected_genes
