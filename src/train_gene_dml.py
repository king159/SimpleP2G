"""Gene-DML with independent validation selection."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time

import anndata
import numpy as np
from omegaconf import OmegaConf
import torch

from src.data.expression import read_panel_counts
from src.baselines.gene_dml_cross.data import GeneDMLSlideCache, restore_mixed_spot_order
from src.baselines.gene_dml_cross.model import GeneDMLCross, GeneDMLCrossConfig
from src.training_support import evaluate_one_step
from src.train_main import (evaluation_rng, rng_state, restore_rng, save_atomic,
                                  validate_config, write_json)


def train(config_path, cache_root, max_epochs=None, smoke=False):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Gene-DML training requires one allocated GPU")
    cfg = OmegaConf.load(config_path)
    validate_config(cfg)
    assert (cfg.total_epochs == 100 or smoke) and cfg.batch_size == 256
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    output = Path(cfg.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_summary.json").exists():
        assert json.loads((output / "run_summary.json").read_text())["status"] == "complete"
        return
    OmegaConf.save(cfg, output / "config.yaml")
    genes = Path(cfg.gene_list_filename).read_text().splitlines()
    ids = list(cfg.training_slides)
    cache_root = Path(cache_root)
    caches, labels = {}, {}

    def load_slide(slide):
        cache = GeneDMLSlideCache(cache_root / slide / "manifest.json")
        a = anndata.read_h5ad(Path(cfg.data_path) / "st" / (slide + ".h5ad"), backed="r")
        try:
            count = read_panel_counts(a, genes)
            count = count.toarray() if hasattr(count, "toarray") else np.asarray(count)
            y = np.log2(np.asarray(count, dtype=np.float32) + 1)
            target = cache.align_expression(y, np.asarray(a.obs_names), np.arange(len(cache)), "cpu")
        finally:
            a.file.close()
        assert torch.isfinite(target).all()
        cache.gpu_global = torch.tensor(np.array(cache.arrays["global_features"]), dtype=torch.float32, device="cuda")
        cache.gpu_positions = torch.tensor(np.array(cache.arrays["positions"]), dtype=torch.long, device="cuda")
        return cache, target

    started = time.perf_counter()
    for slide in ids:
        caches[slide], labels[slide] = load_slide(slide)
    validation_cache, validation_target = load_slide(cfg.validation_slide)
    model_config = GeneDMLCrossConfig(**OmegaConf.to_container(cfg.model_config))
    model = GeneDMLCross(model_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.optimizer.lr),
                                  weight_decay=float(cfg.optimizer.weight_decay), betas=tuple(cfg.optimizer.betas))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg.scheduler.step_size), gamma=float(cfg.scheduler.gamma))
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    pool = np.asarray([(i, spot) for i, slide in enumerate(ids) for spot in range(len(caches[slide]))], dtype=np.int64)
    steps_per_epoch = math.ceil(len(pool) / 256)
    history, best, best_epoch = [], -math.inf, None
    resume = output / "resume.pt"
    config_plain = OmegaConf.to_container(cfg, resolve=True)
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        assert saved["config"] == config_plain
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history, best, best_epoch = saved["history"], saved["best"], saved["best_epoch"]
        rng.bit_generator.state = saved["generator_rng"]
        restore_rng(saved["rng"])
        del saved
    setup_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()

    def load_batch(rows):
        chunks, targets, grouped_rows = [], [], []
        for slide_index in np.unique(rows[:, 0]):
            slide = ids[int(slide_index)]
            cache = caches[slide]
            selected = rows[rows[:, 0] == slide_index, 1]
            grouped_rows.extend(np.flatnonzero(rows[:, 0] == slide_index).tolist())
            chunk = cache.batch(selected, "cpu", view_indices=rng.integers(0, 3, len(selected)), include_global=False)
            chunk["global_features"], chunk["positions"] = cache.gpu_global, cache.gpu_positions
            chunks.append(chunk)
            targets.append(labels[slide][selected])
        inputs = {key: torch.cat([c[key] for c in chunks]).cuda() for key in
                  ("target_tokens", "neighbor_features", "neighbor_mask", "spot_indices")}
        inputs["global_features"] = [c["global_features"] for c in chunks]
        inputs["positions"] = [c["positions"] for c in chunks]
        inputs["slide_indices"] = torch.cat([torch.full((len(c["spot_indices"]),), i, dtype=torch.long)
                                            for i, c in enumerate(chunks)]).cuda()
        return restore_mixed_spot_order(inputs, torch.cat(targets).cuda(), grouped_rows)

    @torch.inference_mode()
    def predict(cache, target, seed):
        model.eval()
        predictions = []
        with evaluation_rng(seed), torch.autocast("cuda", dtype=torch.float16):
            for start in range(0, len(cache), 256):
                inputs = cache.batch(np.arange(start, min(start + 256, len(cache))), "cuda", include_global=False)
                inputs["global_features"], inputs["positions"] = cache.gpu_global, cache.gpu_positions
                predictions.append(model(**inputs)["predictions"]["fusion"].float().cpu().numpy())
        prediction = np.concatenate(predictions)
        assert prediction.shape == target.shape and np.isfinite(prediction).all()
        metrics = evaluate_one_step(prediction, target.numpy())
        if not math.isfinite(metrics["PCC200"]):
            raise ValueError("Non-finite evaluation PCC200")
        return prediction, metrics

    end_epoch = int(cfg.total_epochs) if max_epochs is None else min(int(cfg.total_epochs), len(history) + max_epochs)
    for epoch in range(len(history) + 1, end_epoch + 1):
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        count = skipped = 0
        steps = np.array_split(pool[rng.permutation(len(pool))], steps_per_epoch)
        for step, rows in enumerate(steps, 1):
            inputs, expression = load_batch(rows)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(**inputs, expression=expression)
            if not torch.isfinite(result["loss"]):
                raise ValueError(f"Non-finite loss at epoch {epoch} step {step}")
            scaler.scale(result["loss"]).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < previous_scale
            total_loss += float(result["loss"].detach()) * len(rows)
            count += len(rows)
            del result, inputs, expression
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - started
        scheduler.step()
        started = time.perf_counter()
        _, metrics = predict(validation_cache, validation_target, int(cfg.seed) + 100000)
        validation_seconds = time.perf_counter() - started
        if metrics["PCC200"] > best:
            best, best_epoch = metrics["PCC200"], epoch
            save_atomic(output / "best_model.pt", {"model": model.state_dict(), "epoch": epoch,
                         "validation_metrics": metrics, "weights": "model"})
        row = {"epoch": epoch, "train_loss": total_loss / count, "train_seconds": training_seconds,
               "validation_seconds": validation_seconds, "validation_metrics": metrics,
               "best_epoch": best_epoch, "best_validation_pcc200": best, "steps": len(steps),
               "skipped_steps": skipped, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        save_atomic(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "config": config_plain, "history": history, "best": best, "best_epoch": best_epoch,
                    "rng": rng_state(), "generator_rng": rng.bit_generator.state})
        (output / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
        write_json(output / "status.json", {"state": "training", "outer_test_accessed": False,
                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **row})
        print("EPOCH " + json.dumps(row), flush=True)
    if len(history) < int(cfg.total_epochs):
        return
    selected = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    del selected, validation_cache, validation_target
    caches.clear()
    labels.clear()
    torch.cuda.empty_cache()
    test_cache, target = load_slide(cfg.test_slide)
    started = time.perf_counter()
    prediction, metrics = predict(test_cache, target, int(cfg.seed) + 200000)
    inference_seconds = time.perf_counter() - started
    np.savez_compressed(output / "final_predictions.npz", prediction=prediction, target=target.numpy(),
                         genes=np.asarray(genes), spot_ids=test_cache.arrays["spot_ids"])
    summary = {"status": "complete", "epochs": len(history), "seed": int(cfg.seed),
               "best_epoch": best_epoch, "best_validation_pcc200": best,
               "checkpoint_selection": "best_validation_pcc200", "final_metrics": metrics,
               "test_slide": cfg.test_slide, "validation_slide": cfg.validation_slide,
               "training_slides": list(cfg.training_slides),
               "parameters": sum(p.numel() for p in model.parameters()),
               "train_seconds": sum(r["train_seconds"] for r in history),
               "validation_seconds": sum(r["validation_seconds"] for r in history),
               "setup_seconds": setup_seconds, "inference_seconds": inference_seconds,
               "precision": "fp16", "gpu": torch.cuda.get_device_name(),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(output / "run_summary.json", summary)
    write_json(output / "status.json", dict(summary, state="complete"))
    resume.unlink(missing_ok=True)
    (output / "failure.json").unlink(missing_ok=True)
    print("COMPLETE " + json.dumps(summary), flush=True)
