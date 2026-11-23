# Image encoders and caches

The 300 main fits use frozen Virchow2. Most heads take the 2,560-dimensional concatenation of the CLS token and mean patch token. Gene-DML takes the 261 × 1,280 target token sequence, CLS descriptors for a 5 × 5 neighborhood, and CLS features for the whole slide. Features are cached once per slide and can be reused for each fold. Gene panels are selected separately from each fold’s training slides. See the [extraction instructions](reproducibility.md#features-and-gene-panels).

## Reproduce the encoder comparison

Each of the 120 pairs compares a reference image input to Virchow2 on the same dataset, panel, and fold. The Virchow2 endpoint comes from the main benchmark. Prepare the reference inputs after downloading the slides and selecting the panels.

| Method | Reference input | How to prepare it |
| --- | --- | --- |
| BLEEP | ImageNet ResNet50, trained with the projection heads | The loader reads 224 × 224 RGB patches from each slide. `timm` loads the pretrained ResNet50 when the model is first initialized. |
| HisToGene | Raw 112 × 112 RGB patches | Cache the patches once using the command below. |
| MERGE | 256-dimensional TCGA-pretrained ResNet18 features | Fetch MERGE’s released weights and run the embedding extractor. |
| M2TGLGO | 2,048-dimensional ImageNet ResNet50 features | The same extractor loads torchvision’s `IMAGENET1K_V2` weights. |

Download the [MERGE checkpoint](https://github.com/ags3927/MERGE/blob/614f3fceb359c851d539f3415283320c93ceb5e2/pretrained/model-low-v1.pth) on a machine with network access.

```bash
mkdir -p checkpoints/merge
curl -fL -o checkpoints/merge/model-low-v1.pth \
  https://raw.githubusercontent.com/ags3927/MERGE/614f3fceb359c851d539f3415283320c93ceb5e2/pretrained/model-low-v1.pth
```

Prepare HisToGene’s cache on CPU, then extract the two feature sets on GPU.

```bash
python scripts/cache_histogene_raw_patches.py
python scripts/extract_reference_patch_embeddings.py \
  --data-path hest1k_datasets/{brain,prad,her2st,lung,ccrcc} \
  --encoders resnet50 merge_resnet18 --merge-checkpoint checkpoints/merge/model-low-v1.pth
```

Create the reference configurations in a fresh directory. Each manifest entry gives its training configuration and output path. Run each configuration with `python -m src.train_main --config PATH`, as for the main fits.

```bash
python scripts/build_benchmark.py --inputs reference --output results/reference
```

After both batches finish, evaluate the references against the matching main-run targets. The evaluator checks the ordered gene and spot IDs and confirms that both target arrays represent the same integer counts.

```bash
python scripts/evaluate_results.py --batch results/reference --paired-main results/benchmark
```

The [encoder comparison CSV](results/encoder-upgrades.csv) contains the paired scores, reference inputs and validation epochs. The [column guide](results/README.md) explains how to compute slide-level uncertainty for an encoder change.
