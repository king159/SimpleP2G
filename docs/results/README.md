# Downloadable experimental results

- [Main experiments](main-results.csv): **300 rows**, one fit per model × dataset × panel × fold (10 × 5 × 2 × 3).
- [Image encoder swaps](encoder-upgrades.csv): **120 rows**, one matched reference/Virchow2 pair per method × dataset × panel × fold (4 × 5 × 2 × 3). The Virchow2 fits are reused from the main experiments.

The paper and project page aggregate these unrounded, per-fold values. Each
dataset–panel–method group has folds 0, 1, and 2, all with seed 42. Report the
arithmetic mean and **sample SD (`ddof=1`) across slides**. These are slide-level
holdouts, not three random seeds or necessarily three independent patients.

## Columns

Both files identify `dataset`, `panel` (HMHVG/HVG, 200 genes), `regime`
(`single`/`cross`), `method`, `adapted`, `fold`, `seed`, `test_slide`, and
`validation_slide`. They include the epoch budget and validation-selected epoch.
Stem trains for 200 epochs; other methods train for 100.

`pcc200` and `pcc50` are higher-is-better; `mse` and `mae` are lower-is-better.
Targets use log₂(raw count + 1). PCC excludes genes with constant observed
expression; a constant prediction for a variable target receives zero
correlation. PCC-50 averages the highest 50 evaluable per-gene correlations.
The `adapted` flag identifies STRank and M2TGLGO variants documented in the paper.
STRank's raw ranking outputs are not calibrated expression estimates.

The encoder file provides `reference_*`, `virchow2_*`, and `delta_*` for each
metric. Every delta is **Virchow2 minus reference on the same fold**, before
rounding. For uncertainty in the encoder effect, compute the sample SD of
`delta_pcc200`; it is distinct from either endpoint's SD. Reference inputs are
specified per row. All paired fits use the same split, gene panel, and test spots.

`evaluation_precision` fields record the evaluation mode. Two BLEEP reference
fits use FP32 evaluation after resumption: PRAD/HVG/fold 0 from epoch 99 and
ccRCC/HVG/fold 2 from epoch 87. Their earlier validation scores remain the recorded
FP16 scores; final test inference is FP32. The column
`reference_validation_fp32_from_epoch` records these transitions and is otherwise
empty.

The SHA-256 fields identify the ordered gene and spot lists, target array, and saved prediction file for each fit.

Metrics use the same target array within each dataset/panel/fold. Tiny float32
log₂ differences in some reference archives are checked against the same integer
counts before using that common array. `reference_stored_target_sha256` and
`reference_target_max_abs_difference` preserve this distinction; saved predictions
are unchanged.

## Aggregate a download

```python
from pathlib import Path

import pandas as pd

# Run from the repository root.
results = Path("docs/results")
main = pd.read_csv(results / "main-results.csv")
summary = main.groupby(["regime", "dataset", "panel", "method"])[
    ["pcc200", "pcc50", "mse", "mae"]
].agg(["mean", "std"])

pairs = pd.read_csv(results / "encoder-upgrades.csv")
encoder_effect = pairs.groupby(["dataset", "panel", "method"])[
    ["reference_pcc200", "virchow2_pcc200", "delta_pcc200"]
].agg(["mean", "std"])
```
