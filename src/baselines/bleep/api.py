"""BLEEP integration helpers.

Goal: keep `src/train_main.py` free of BLEEP-specific branching.

This module provides:
- model-name predicates
- model builder
- training loss adapter
- evaluation/inference (retrieval from a bank)

It supports both:
- `bleep` / `bleep_enhanced`: uses precomputed image embeddings as condition.
- `bleep_original`: follows the original implementation (timm ResNet50 on pixels).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.baselines.bleep.model import BleepModel, build_bleep_from_omegaconf
from src.baselines.bleep_original.model import BleepOriginalModel, build_bleep_original_from_omegaconf


def _unwrap(model):
    """Unwrap torch.compile + DDP/FSDP wrappers to access custom methods."""
    m = model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    if hasattr(m, "module"):
        m = m.module
    return m


_BLEEP_EMBEDDING_NAMES = {"bleep", "bleep_enhanced"}
_BLEEP_ORIGINAL_NAME = "bleep_original"


def is_bleep_model(model_name: str) -> bool:
    return str(model_name).lower() in (_BLEEP_EMBEDDING_NAMES | {_BLEEP_ORIGINAL_NAME})


def is_bleep_original(model_name: str) -> bool:
    return str(model_name).lower() == _BLEEP_ORIGINAL_NAME


def build_bleep_any_from_omegaconf(args) -> BleepModel | BleepOriginalModel:
    model_name = str(getattr(args, "model", "")).lower()
    if model_name in _BLEEP_EMBEDDING_NAMES:
        return build_bleep_from_omegaconf(args)
    if model_name == _BLEEP_ORIGINAL_NAME:
        return build_bleep_original_from_omegaconf(args)
    raise ValueError(f"Not a BLEEP model: {model_name}")


def bleep_compute_loss(model_name: str, model, *, x: torch.Tensor, y, args) -> torch.Tensor:
    """Adapter used in the main training loop.

    Args:
      model_name: args.model.lower()
      model: prepared model (may be DDP/compiled)
      x: gene vector (B, G)
      y: either embedding (B, D) or dict containing `image`
    """

    raw_model = _unwrap(model)
    if is_bleep_original(model_name):
        image = y["image"]
        return raw_model.compute_loss(image, x)

    # embedding mode: y is already image embeddings
    return raw_model.compute_loss(y, x)


@torch.no_grad()
def bleep_generate_samples(
    model_name: str, *, eval_loader, accelerator, model, train_set, args
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate expression predictions for eval_loader.

    Returns:
      pred: (N, G) numpy
      test: (N, G) numpy
    """

    bleep_cfg = getattr(args, "bleep", None)
    topk = int(getattr(bleep_cfg, "topk", 64)) if bleep_cfg is not None else 64

    # Original modality: build bank via DataLoader (needs images)
    if is_bleep_original(model_name):
        bank_bs = int(getattr(getattr(args, "bleep", None), "bank_batch_size", 64))
        bank_loader = DataLoader(
            train_set,
            batch_size=bank_bs,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        raw_model = _unwrap(model)
        bank_image_z, bank_x = raw_model.build_image_bank(bank_loader)

        all_samples = []
        all_true = []
        for x, y in eval_loader:
            x = x.to(accelerator.device)
            image = y["image"].to(accelerator.device)
            pred = raw_model.predict_from_bank(image, bank_image_z, bank_x, topk=topk)
            all_samples.append(pred.detach().cpu())
            all_true.append(x.detach().cpu())

        pred_np = torch.cat(all_samples, dim=0).numpy()
        test_np = torch.cat(all_true, dim=0).numpy()
        return pred_np, test_np

    # Embedding mode: bank is (bank_x gene vec, bank_y embedding)
    bank_subsample = int(getattr(bleep_cfg, "bank_subsample", -1)) if bleep_cfg is not None else -1
    bank_batch_size = int(getattr(bleep_cfg, "bank_batch_size", 4096)) if bleep_cfg is not None else 4096

    bank_x = train_set.data
    bank_y = train_set.label

    if bank_subsample is not None and bank_subsample > 0 and bank_subsample < len(bank_x):
        g = torch.Generator().manual_seed(int(getattr(args, "seed", 42)))
        idx = torch.randperm(len(bank_x), generator=g)[:bank_subsample]
        bank_x = bank_x[idx]
        bank_y = bank_y[idx]

    bank_x = bank_x.to(accelerator.device)
    bank_y = bank_y.to(accelerator.device)
    raw_model = _unwrap(model)
    bank_image_z = raw_model.build_image_bank(bank_y, batch_size=bank_batch_size)

    all_samples = []
    all_true = []
    for x, y in eval_loader:
        y = y.to(accelerator.device)
        pred = raw_model.predict_from_bank(y, bank_image_z, bank_x, topk=topk)
        all_samples.append(pred.detach().cpu())
        all_true.append(x.detach().cpu())

    pred_np = torch.cat(all_samples, dim=0).numpy()
    test_np = torch.cat(all_true, dim=0).numpy()
    return pred_np, test_np


def train_one_step(model, batch, args, **kwargs):
    model_name = str(getattr(args, "model", "")).lower()
    if isinstance(batch, dict):
        x, y = None, None
    else:
        x, y = batch
    return bleep_compute_loss(model_name, model, x=x, y=y, args=args)


@torch.no_grad()
def eval_one_step(model, eval_dataloader, accelerator, args, train_set=None, **kwargs):
    model_name = str(getattr(args, "model", "")).lower()
    return bleep_generate_samples(
        model_name,
        eval_loader=eval_dataloader,
        accelerator=accelerator,
        model=model,
        train_set=train_set,
        args=args,
    )
