# Benchmark configurations

Each file in [models/](models/) contains the editable defaults for one prediction model. The ten files named for the methods use Virchow2. Four additional files with the `_reference.yaml` suffix specify the paired reference inputs for BLEEP, HisToGene, MERGE, and M2TGLGO. Both gene panels use the same defaults. The `cohort_overrides` section in [simplep2g.yaml](models/simplep2g.yaml) records the dataset-specific dropout, warmup steps, and optimizer settings.

The [benchmark protocol](protocols/benchmark.yaml) specifies training budgets and precision transitions. The [split manifest](splits/heldout.json) lists the training, validation, and test slides for all 15 folds. After preparing the gene panels, generate the run configurations from the repository root.

```bash
python scripts/build_benchmark.py --output results/benchmark
python scripts/build_benchmark.py --inputs reference --output results/reference
```

The first command generates 300 Virchow2 configurations. The second generates 120 paired reference configurations with matching splits, panels, and seeds. To prepare a smaller example, add `--cohort brain --fold 0 --method simplep2g` to the first command. Each run uses the checkpoint with the highest validation PCC-200. See the [reproduction guide](../docs/reproducibility.md) for the training commands.
