"""Verify saved predictions and write per-run and three-slide aggregate CSV files."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training_support import evaluate_one_step


def reported_metrics(prediction, target):
    """PCC ranks variable targets and assigns zero to constant predictions."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    variable = np.ptp(target, axis=0) > 0
    if not variable.any():
        raise ValueError("The held-out slide has no variable target genes")
    x = target[:, variable] - target[:, variable].mean(axis=0)
    y = prediction[:, variable] - prediction[:, variable].mean(axis=0)
    divisor = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0))
    correlations = np.zeros(variable.sum())
    np.divide((x * y).sum(axis=0), divisor, out=correlations,
              where=np.ptp(prediction[:, variable], axis=0) > 0)
    ranked = np.sort(correlations)[::-1]
    errors = prediction - target
    return {f"PCC{k}": float(ranked[:k].mean()) for k in (10, 50, 200)} | {
        "MSE": float(np.mean(errors**2)), "MAE": float(np.mean(np.abs(errors))),
    }


def pair_key(task):
    return tuple(task[field] for field in ("method", "cohort", "panel", "fold", "test_slide"))


def collect(batch, paired_main=None):
    import pandas as pd
    manifest = json.loads((batch / "manifest.json").read_text())
    paired_outputs = {}
    if paired_main is not None:
        main_manifest = json.loads((paired_main / "manifest.json").read_text())
        paired_outputs = {pair_key(task): paired_main / task["output"] / "final_predictions.npz"
                          for task in main_manifest["tasks"]}
    rows = []
    for task in manifest["tasks"]:
        folder = batch / task["output"]
        summary = json.loads((folder / "run_summary.json").read_text())
        if summary["status"] != "complete" or summary["checkpoint_selection"] != "best_validation_pcc200":
            raise ValueError(f"Incomplete or unexpected protocol: {folder}")
        with np.load(folder / "final_predictions.npz", allow_pickle=False) as saved:
            pred, target = saved["prediction"], saved["target"]
            if pred.shape != target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
                raise ValueError(f"Invalid prediction arrays: {folder}")
            if pred.shape != (len(saved["spot_ids"]), len(saved["genes"])):
                raise ValueError(f"Prediction identifiers do not align: {folder}")
            recorded = evaluate_one_step(pred, target)
            if paired_main is not None:
                with np.load(paired_outputs[pair_key(task)], allow_pickle=False) as paired:
                    np.testing.assert_array_equal(saved["genes"], paired["genes"])
                    np.testing.assert_array_equal(saved["spot_ids"], paired["spot_ids"])
                    common_target = paired["target"]
                    np.testing.assert_allclose(target, common_target, rtol=0, atol=2e-6)
                    np.testing.assert_array_equal(np.rint(np.exp2(target.astype(np.float64)) - 1),
                                                  np.rint(np.exp2(common_target.astype(np.float64)) - 1))
                    target = common_target
            metrics = reported_metrics(pred, target)
        for metric, value in recorded.items():
            np.testing.assert_allclose(value, summary["final_metrics"][metric], rtol=1e-6, atol=1e-7)
        rows.append({k: task[k] for k in ("method", "cohort", "panel", "fold", "test_slide")} |
                    {"best_epoch": summary["best_epoch"], **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(batch / "per_run.csv", index=False, float_format="%.17g")
    grouped = frame.groupby(["method", "cohort", "panel"])[["PCC200", "PCC50", "MSE", "MAE"]]
    aggregate = grouped.agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(c) for c in aggregate.columns]
    aggregate.to_csv(batch / "by_dataset.csv", float_format="%.17g")
    print(f"Verified {len(frame)} runs; wrote per_run.csv and by_dataset.csv (sample SD, ddof=1)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--paired-main", type=Path, help="Match each reference run to its Virchow2 labels")
    args = parser.parse_args()
    collect(args.batch.resolve(), args.paired_main)
