"""Independent slide validation; the outer test is constructed only after selection."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import anndata
from accelerate import Accelerator
from accelerate.utils import set_seed
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from src.training_support import (CROSS_PATCH_SIMPLE_MODELS, CustomDataset,
    _assemble_neigh_slide_dataset, _assemble_slide_embedding_dataset,
    _load_slide_embeddings, customized_lr_scheduler, evaluate_one_step, load_train_objs)
from src.train_utils import create_logger, update_ema


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def evaluation_rng(seed):
    """Evaluation must neither alter training shuffles nor reward lucky changing noise."""
    state = rng_state()
    set_seed(seed)
    try:
        yield
    finally:
        restore_rng(state)


def validate_config(cfg):
    if cfg.get("config_stage") != "benchmark":
        raise ValueError("Materialize a benchmark configuration with scripts/build_benchmark.py")
    obsolete = {"select_best_on_eval", "ckpt_every", "save_eval_artifacts", "head_type"} & set(cfg)
    if obsolete:
        raise ValueError(f"Unsupported configuration fields: {sorted(obsolete)}")
    train = list(cfg.training_slides)
    validation, test = str(cfg.validation_slide), str(cfg.test_slide)
    if len(set(train)) != len(train) or not train:
        raise ValueError("Training slides must be nonempty and distinct")
    if validation == test or {validation, test} & set(train):
        raise ValueError("Training, validation and test slides overlap")
    if cfg.slide_out != validation:
        raise ValueError("The training loader's evaluation slide must be validation")
    expected = set(train) | {validation}
    actual = Path(cfg.folder_list_filename).read_text().splitlines()
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Training loader may see only training and validation slides")
    if cfg.checkpoint_selection != "best_validation_pcc200":
        raise ValueError("Unsupported checkpoint rule")


def panel_counts(data_path, slide, genes):
    from src.data.expression import read_panel_counts
    a = anndata.read_h5ad(Path(data_path) / "st" / f"{slide}.h5ad", backed="r")
    try:
        x = read_panel_counts(a, genes)
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        ids = np.asarray(a.obs_names, dtype=str)
    finally:
        a.file.close()
    return np.asarray(x, dtype=np.float32), ids


def evaluation_dataset(cfg, slide, logger):
    """Construct exactly one evaluation slide, with no auxiliary training loads."""
    genes = Path(cfg.gene_list_filename).read_text().splitlines()
    args = deepcopy(cfg)
    args.slide_out = slide
    name = str(cfg.model).lower()
    if name in CROSS_PATCH_SIMPLE_MODELS:
        return _assemble_neigh_slide_dataset(args, logger, [], genes)[1]
    if name in {"stflow", "merge", "m2tglgo"}:
        return _assemble_slide_embedding_dataset(args, logger, [], genes)[1]
    if name in {"histogene", "histogene_enhanced", "histogene_original"}:
        from src.data.histogene_dataset import HisToGeneSlideDataset
        from src.data.hest_wsi_dataset import WSIPatchConfig
        use_embeddings = name != "histogene_original"
        encoder_args = {"encoder_name": cfg.image_encoder} if use_embeddings else {}
        return HisToGeneSlideDataset(slide_ids=[slide], data_path=cfg.data_path,
               selected_genes=genes, use_embeddings=use_embeddings, **encoder_args,
               patch_cfg=WSIPatchConfig(patch_size=int(cfg.get("patch_size", 112)),
                   level=int(cfg.get("wsi_level", 0)), imagenet_norm=bool(cfg.get("imagenet_norm", True))),
               n_pos_bins=int(getattr(getattr(cfg, "histogene", None), "n_pos", 64)), train=False)
    if name == "bleep_original":
        from src.data.hest_wsi_dataset import HESTWSISpotDataset, WSIPatchConfig
        return HESTWSISpotDataset(slide_ids=[slide], data_path=cfg.data_path,
            selected_genes=genes, train=False, n_pos_bins=64,
            patch_cfg=WSIPatchConfig(patch_size=int(cfg.get("patch_size", 224)),
                level=int(cfg.get("wsi_level", 0)), imagenet_norm=bool(cfg.get("imagenet_norm", True))))
    counts, _ = panel_counts(cfg.data_path, slide, genes)
    features, _ = _load_slide_embeddings(cfg.data_path, slide, cfg.image_encoder, load_aug=False)
    if name == "strank":
        from src.baselines.strank.data import STRankSpotDataset
        return STRankSpotDataset(counts, features, np.zeros(len(counts), dtype=np.int64))
    return CustomDataset(torch.from_numpy(np.log2(counts + 1)), features.float())


def api_for(cfg):
    name = str(cfg.model).lower()
    extra = {}
    if name == "simplep2g_cross":
        module = "ours.api_neigh"
    elif name in {"bleep_enhanced", "bleep", "bleep_original"}:
        module = "bleep.api"
    elif name in {"histogene", "histogene_enhanced", "histogene_original"}:
        module = "histogene.api"
    else:
        module = name + ".api"
    if name == "stem":
        from src.baselines.stem.diffusion import create_diffusion
        extra["diffusion"] = create_diffusion(timestep_respacing="")
    elif name == "stflow":
        from src.baselines.stflow.api import build_interpolant_from_omegaconf
        extra["interpolant"] = build_interpolant_from_omegaconf(cfg)
    return importlib.import_module("src.baselines." + module), extra


def train(config_path, *, allow_cpu=False, max_epochs=None, smoke=False):
    cfg = OmegaConf.load(config_path)
    validate_config(cfg)
    if not allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    budget = 200 if str(cfg.model).lower() == "stem" else 100
    if not smoke and int(cfg.total_epochs) != budget:
        raise ValueError(f"The benchmark budget for {cfg.model} is {budget} epochs")
    output = Path(cfg.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_summary.json").exists():
        if json.loads((output / "run_summary.json").read_text())["status"] == "complete":
            return
    name = cfg.model.lower()
    precision = "no" if allow_cpu or name == "strank" else "fp16"
    if cfg.get("bf16_from_epoch") and not allow_cpu:
        # Accelerate fixes precision per process. Run the FP16 prefix in a
        # separate process when this invocation spans the recorded boundary.
        resume = output / "resume.pt"
        completed = (len(torch.load(resume, map_location="cpu", weights_only=False, mmap=True)["history"])
                     if resume.exists() else 0)
        fp16_epochs_left = int(cfg.bf16_from_epoch) - 1 - completed
        if fp16_epochs_left <= 0:
            precision = "bf16"
        elif max_epochs is None or max_epochs > fp16_epochs_left:
            subprocess.run([sys.executable, "-m", "src.train_main", "--config", str(config_path),
                            "--max-epochs", str(fp16_epochs_left)], check=True)
            if max_epochs is not None:
                max_epochs -= fp16_epochs_left
            precision = "bf16"
    logger = create_logger(str(output))
    OmegaConf.save(cfg, output / "config.yaml")
    accelerator = Accelerator(mixed_precision=precision, cpu=allow_cpu)
    if accelerator.num_processes != 1:
        raise ValueError("Each fit must use exactly one GPU/process")
    set_seed(int(cfg.seed))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    start = time.perf_counter()
    train_set, validation_set, model, cfg = load_train_objs(cfg, logger)
    slide_level = name in {"histogene", "histogene_enhanced", "histogene_original", "stflow", "merge", "m2tglgo", "simplep2g_cross"}
    size = 1 if slide_level else int(cfg.per_gpu_batch_size)
    loader = DataLoader(train_set, batch_size=size, shuffle=True, num_workers=0,
                        pin_memory=not allow_cpu, drop_last=not slide_level)
    validation_loader = DataLoader(validation_set, batch_size=size, shuffle=False,
                                   num_workers=0, pin_memory=not allow_cpu)
    if not len(loader):
        raise ValueError("No training batches")
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    opt_class = optimizers[str(cfg.optimizer.name).lower()]
    optimizer = opt_class(model.parameters(), lr=float(cfg.optimizer.lr),
                          weight_decay=float(cfg.optimizer.weight_decay), betas=tuple(cfg.optimizer.betas))
    scheduler = customized_lr_scheduler(optimizer, warmup_steps=int(cfg.get("warmup_steps", 0)),
                  total_steps=int(cfg.total_epochs) * len(loader) if cfg.get("cosine_decay", False) else -1,
                  min_lr_ratio=float(cfg.get("min_lr_ratio", 0.1)))
    ema = deepcopy(model).to(accelerator.device) if float(cfg.get("ema_decay", 0)) > 0 else None
    if ema is not None:
        ema.requires_grad_(False)
    model, optimizer, loader, validation_loader = accelerator.prepare(model, optimizer, loader, validation_loader)
    api, extra = api_for(cfg)
    history, best, best_epoch = [], -math.inf, None
    resume = output / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        assert state["config"] == OmegaConf.to_container(cfg, resolve=True)
        accelerator.unwrap_model(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        if accelerator.scaler is not None and state["scaler"]:
            accelerator.scaler.load_state_dict(state["scaler"])
        history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
        restore_rng(state["rng"])
        del state
    setup_seconds = time.perf_counter() - start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def evaluate(data, phase, epoch):
        evaluation_model = ema if ema is not None else model
        evaluation_model.eval()
        eval_cfg = deepcopy(cfg)
        if phase == "validation" and name == "stem":
            eval_cfg.num_eval_samples = int(cfg.get("validation_num_eval_samples", 1))
        fp32_from = cfg.get("validation_fp32_from_epoch")
        use_fp32 = fp32_from is not None and epoch >= fp32_from
        with evaluation_rng(int(cfg.seed) + (100000 if phase == "validation" else 200000)):
            with (torch.autocast(accelerator.device.type, enabled=False) if use_fp32 else accelerator.autocast()):
                prediction, target = api.eval_one_step(evaluation_model, data, accelerator, eval_cfg,
                                                       train_set=train_set, **extra)
        prediction, target = np.asarray(prediction, dtype=np.float32), np.asarray(target, dtype=np.float32)
        if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f"Invalid {phase} predictions")
        metrics = evaluate_one_step(prediction, target)
        if not math.isfinite(metrics["PCC200"]):
            raise ValueError(f"Non-finite {phase} PCC200")
        return prediction, target, metrics

    end_epoch = int(cfg.total_epochs) if max_epochs is None else min(int(cfg.total_epochs), len(history) + max_epochs)
    for epoch in range(len(history) + 1, end_epoch + 1):
        model.train()
        started = time.perf_counter()
        losses, skips = [], 0
        for index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = api.train_one_step(model, batch, cfg, **extra)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch} batch {index}")
            accelerator.backward(loss)
            if cfg.get("max_grad_norm") is not None:
                accelerator.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
            optimizer.step()
            skipped = bool(accelerator.optimizer_step_was_skipped)
            skips += skipped
            if not skipped:
                scheduler.step()
                if ema is not None:
                    update_ema(ema, accelerator.unwrap_model(model), decay=float(cfg.ema_decay))
            losses.append(float(loss.detach()))
            del loss, batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        gc.collect()
        started = time.perf_counter()
        _, _, metrics = evaluate(validation_loader, "validation", epoch)
        validation_seconds = time.perf_counter() - started
        # Strict improvement gives the prespecified earliest-epoch tie break.
        if metrics["PCC200"] > best:
            best, best_epoch = metrics["PCC200"], epoch
            chosen = ema if ema is not None else accelerator.unwrap_model(model)
            save_atomic(output / "best_model.pt", {"model": chosen.state_dict(), "epoch": epoch,
                        "validation_metrics": metrics, "weights": "ema" if ema is not None else "model"})
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_metrics": metrics,
               "train_seconds": training_seconds, "validation_seconds": validation_seconds,
               "best_epoch": best_epoch, "best_validation_pcc200": best,
               "lr": float(optimizer.param_groups[0]["lr"]), "skipped_steps": skips}
        history.append(row)
        save_atomic(resume, {"model": accelerator.unwrap_model(model).state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                    "history": history, "best": best, "best_epoch": best_epoch, "rng": rng_state(),
                    "config": OmegaConf.to_container(cfg, resolve=True)})
        (output / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        write_json(output / "status.json", {"state": "training", "outer_test_accessed": False,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **row})
        print("EPOCH " + json.dumps(row), flush=True)
    if len(history) < int(cfg.total_epochs):
        return  # Verification stop: resume later without touching the test slide.
    selected = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    chosen = ema if ema is not None else accelerator.unwrap_model(model)
    chosen.load_state_dict(selected["model"])
    del selected
    # This is the first outer-test data construction in the run.
    test = evaluation_dataset(cfg, cfg.test_slide, logger)
    test_loader = accelerator.prepare(DataLoader(test, batch_size=size, shuffle=False, num_workers=0))
    started = time.perf_counter()
    prediction, target, metrics = evaluate(test_loader, "test", len(history))
    inference_seconds = time.perf_counter() - started
    genes = Path(cfg.gene_list_filename).read_text().splitlines()
    counts, spot_ids = panel_counts(cfg.data_path, cfg.test_slide, genes)
    np.testing.assert_allclose(target, np.log2(counts + 1), rtol=1e-6, atol=1e-6)
    np.savez_compressed(output / "final_predictions.npz", prediction=prediction, target=target,
                        genes=np.asarray(genes), spot_ids=spot_ids)
    summary = {"status": "complete", "epochs": len(history), "seed": int(cfg.seed),
               "best_epoch": best_epoch, "best_validation_pcc200": best,
               "checkpoint_selection": "best_validation_pcc200", "final_metrics": metrics,
               "test_slide": cfg.test_slide, "validation_slide": cfg.validation_slide,
               "training_slides": list(cfg.training_slides), "parameters": sum(p.numel() for p in model.parameters()),
               "train_seconds": sum(r["train_seconds"] for r in history),
               "validation_seconds": sum(r["validation_seconds"] for r in history),
               "setup_seconds": setup_seconds, "inference_seconds": inference_seconds,
               "precision": precision, "evaluation_weights": "ema" if ema is not None else "model",
               "evaluation_precision": "fp32" if cfg.get("validation_fp32_from_epoch") else precision,
               "validation_fp32_from_epoch": cfg.get("validation_fp32_from_epoch"),
               "bf16_from_epoch": cfg.get("bf16_from_epoch"),
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(output / "run_summary.json", summary)
    write_json(output / "status.json", dict(summary, state="complete"))
    if name == "stem":
        resume.replace(output / "last_training.pt")
    else:
        resume.unlink(missing_ok=True)
    (output / "failure.json").unlink(missing_ok=True)
    print("COMPLETE " + json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--cache-root", type=Path, help="Gene-DML image cache for this cohort")
    args = parser.parse_args()
    if OmegaConf.load(args.config).model == "gene_dml_cross":
        if args.cache_root is None:
            parser.error("Gene-DML requires --cache-root")
        from src.train_gene_dml import train as train_gene_dml
        train_gene_dml(args.config, args.cache_root, max_epochs=args.max_epochs)
    else:
        train(args.config, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
