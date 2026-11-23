import numpy as np
import torch

from src.simpleP2G import get_loss_fn


def train_one_step(model, batch, args, **kwargs):
    x, y = batch
    if args.loss_name in ["gaussian_nll", "beta_nll"]:
        mean_preds, log_var_pred = model(y)
        loss = get_loss_fn(args.loss_name)(mean_preds, log_var_pred, x)
    elif args.loss_name in ["mse", "l1"]:
        mean_preds, log_var_pred = model(y)
        loss = get_loss_fn(args.loss_name)(mean_preds, x)
    return loss


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, **kwargs):
    all_samples = []
    test_count_mtx_selected_genes = []
    for x, y in eval_dataloader:
        samples = model(y)
        if isinstance(samples, tuple):
            samples = samples[0]
        all_samples.append(samples.detach().cpu())
        test_count_mtx_selected_genes.append(x.cpu().numpy())
    pred = torch.cat(all_samples, dim=0).numpy()
    test_count_mtx_selected_genes = np.concatenate(test_count_mtx_selected_genes, axis=0)
    return pred, test_count_mtx_selected_genes
