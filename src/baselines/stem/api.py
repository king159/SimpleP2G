import numpy as np
import torch


def train_one_step(model, batch, args, diffusion=None, **kwargs):
    x, y = batch
    x = x.unsqueeze(1)
    t = torch.randint(0, diffusion.num_timesteps, (x.size(0),), device=x.device)
    model_kwargs = {"y": y}
    loss_dict = diffusion.training_losses(model, x, t, model_kwargs or {})
    loss = loss_dict["loss"].mean()
    return loss


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, diffusion=None, **kwargs):
    num_samples = int(getattr(args, "num_eval_samples", 20))
    all_samples = []
    test_count_mtx_selected_genes = []
    for x, y in eval_dataloader:
        y = y.to(accelerator.device)
        batch_preds = []
        for _ in range(num_samples):
            z = torch.randn(y.shape[0], 1, args.input_gene_size, device=accelerator.device)
            model_kwargs = {"y": y}
            samples = diffusion.p_sample_loop(
                model.forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=accelerator.device,
                progress=False,
            )
            batch_preds.append(samples.squeeze(1))
        avg_pred = torch.stack(batch_preds, dim=0).mean(dim=0)
        all_samples.append(avg_pred.detach().cpu())
        test_count_mtx_selected_genes.append(x.cpu().numpy())
    pred = torch.cat(all_samples, dim=0).numpy()
    test_count_mtx_selected_genes = np.concatenate(test_count_mtx_selected_genes, axis=0)
    return pred, test_count_mtx_selected_genes
