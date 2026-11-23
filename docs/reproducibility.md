# Reproduce the benchmark

Install the [Python environment](../README.md#installation) and run commands from the repository root. The fixed manifest contains three
slide-level splits per cohort, with a disjoint validation slide in each split.
All three test folds use seed 42. The reported SD is the sample SD across their test slides.

## Data and weights

Obtain access to [HEST](https://huggingface.co/datasets/MahmoodLab/hest) and [Virchow2](https://huggingface.co/paige-ai/Virchow2), then authenticate with Hugging Face. Download on
a networked machine into shared storage:

```bash
hf auth login
python scripts/download_data.py plan
python scripts/download_data.py download --dataset all --workers 3
python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download('paige-ai/Virchow2',
    revision='3158645804b69e3f3bc4439d4116edddf0840a72',
    local_dir='checkpoints/Virchow2',
    allow_patterns=['config.json', 'model.safetensors'])
PYCODE
```

For the pretrained Brain inference example alone, use `download --dataset brain --slide NCBI667` in place of the all-cohort download. For a Brain training run, download the entire Brain cohort with `download --dataset brain`.

The planner fetches `HEST_v1_2_1.csv` from the gated HEST repository at revision `7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec`. It uses that metadata to select the five cohorts. The downloader uses the same revision for the slide images, expression matrices, and slide metadata, and verifies the files against the manifest.

## Features and gene panels

On a GPU, extract frozen Virchow2 embeddings for each available slide. For Brain:

```bash
for expression in hest1k_datasets/brain/st/*.h5ad; do
  slide=$(basename "$expression" .h5ad)
  python scripts/prepare_data.py features --cohort brain --slide "$slide" \
    --model-dir checkpoints/Virchow2 --batch-size 64
done
```

This produces a pooled Virchow2 descriptor and seven deterministic augmented
views per spot in expression-file order. Gene-DML uses a separate cache:

```bash
python scripts/extract_gene_dml_features.py \
  --data-path hest1k_datasets/brain --slides NCBI667 \
  --model-dir checkpoints/Virchow2 --cache-root results/features/gene_dml/brain \
  --target-boundary reject
```

Repeat for every slide, including validation and test slides. This operation
uses images and spatial metadata, not expression labels. Use the explicit
`--target-boundary pad` policy for Lung, whose target crops can cross the image
boundary. Other cohorts use `reject`. Each cache records the boundary policy.

On a CPU node, prepare panels for all 15 splits:

```bash
for cohort in brain prad her2st lung ccrcc; do
  for fold in 0 1 2; do
    python scripts/prepare_data.py panels --cohort "$cohort" --fold "$fold"
  done
done
```

Panel preparation refuses to overwrite an existing destination. For each split,
HMHVG and HVG selection use only training-slide expression values. Validation and
test files are opened in backed mode only to inspect their gene identifiers.
The output includes the selected gene lists, training/validation slide list,
provenance and checksums.

## Training and checkpoint selection

```bash
python scripts/build_benchmark.py --output results/benchmark
python -m src.train_main \
  --config results/benchmark/configs/brain_fold0_hmhvg_simplep2g_seed42.yaml
python -m src.train_main \
  --config results/benchmark/configs/brain_fold0_hmhvg_gene_dml_seed42.yaml \
  --cache-root results/features/gene_dml/brain
```

The builder’s default covers ten methods, five datasets, two panels and three folds (300 configurations). SimpleP2G consumes the target patch. SimpleP2G-cross adds the 24 nearest spatial neighbors. STRank and M2TGLGO use benchmark adaptations documented in the [upstream notices](../THIRD_PARTY_NOTICES.md). Optional `--cohort`, `--fold` and
`--method` filters build a smaller subset. Run each generated configuration with
one process on one GPU; select the device with `CUDA_VISIBLE_DEVICES` when needed.
For the reference image inputs, follow the [encoder comparison workflow](encoders.md#reproduce-the-encoder-comparison).

Stem trains for 200 epochs. The other nine methods train for 100. The checkpoint with the highest validation PCC-200 is selected, with ties resolved in favor of the earliest epoch. Validation randomness is fixed and isolated
from training. Stem uses one diffusion sample per spot during validation and
averages 20 at test time. Test expression is loaded only after checkpoint
selection.

Most runs use FP16 mixed precision. STRank uses FP32. Three HMHVG Stem fits
switch from FP16 to BF16 during training, starting at epoch 192 on ccRCC fold 1,
epoch 130 on PRAD fold 1, and epoch 158 on Brain fold 2. Two BLEEP reference
fits switch validation and test inference to FP32, starting at epoch 99 on PRAD
HVG fold 0 and epoch 87 on ccRCC HVG fold 2. The earlier validation scores
remain eligible for checkpoint selection. The downloadable tables record the
[evaluation precision of each fit](results/README.md).

Re-running the same command resumes an interrupted fit from `resume.pt`; a
completed run is left intact. `--max-epochs N` can stop a fit after N additional
epochs without accessing the test slide. It does not change the total budget.

## Evaluation

Each fit writes its resolved configuration to `config.yaml`, training and validation metrics to `history.jsonl`, and the selected checkpoint to `best_model.pt`. Final test predictions, targets, ordered gene IDs and spot IDs are saved in `final_predictions.npz`. The selected epoch and test metrics are in `run_summary.json`. Stem also saves its last training state in `last_training.pt`.

After every fit in the generated batch completes, run

```bash
python scripts/evaluate_results.py --batch results/benchmark
```

The evaluator recomputes PCC-200, MSE and MAE from `final_predictions.npz`, checks
the recorded metrics, and writes `per_run.csv` and `by_dataset.csv`. Aggregate
files report means, sample SDs and counts, so a deliberately selected subset is
not mistaken for a complete three-slide result. Predictions use log₂(count+1)
labels; STRank is supervised with raw counts and converts its evaluation target
to the common scale. Its ranking scores are not calibrated expression values.
