<h1 align="center">SimpleP2G: Importance of Image Encoders for Gene Expression Prediction from Pathology Images</h1>

<p align="center">
  <a href="https://personal-page.wjh.app/">Jinghao Wang</a><sup>1</sup>,
  <a href="https://zhangjj26.github.io/">Jingjie Zhang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=K-AjhSgAAAAJ">Chang-Yu Hsieh</a><sup>2</sup>,<br>
  <a href="https://scholar.google.com/citations?user=BlSFT8UAAAAJ">Chunbin Gu</a><sup>1,✉</sup>,
  <a href="https://appsrv.cse.cuhk.edu.hk/~pheng/index.php">Pheng-Ann Heng</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> The Chinese University of Hong Kong &nbsp;
  <sup>2</sup> Zhejiang University &nbsp;
  <sup>✉</sup> <a href="mailto:gchb4science@gmail.com">Corresponding author</a>
</p>

<div align="center">
  <a href="https://huggingface.co/king159/SimpleP2G">🤗 Pretrained Models</a> ·
  <a href="docs/results/main-results.csv">Main Results CSV</a> ·
  <a href="docs/results/encoder-upgrades.csv">Encoder Swaps CSV</a>
</div>

> **TL;DR:** Improve the image representation before adding a more complex prediction head. Across five datasets and two gene panels, a direct gene-query decoder on frozen pathology embeddings provides a strong baseline for predicting expression from tissue images.

<p align="center">
  <img src="docs/teaser.svg" alt="SimpleP2G overview: single-patch and cross-patch gene expression prediction, with gene queries attending to frozen pathology image features." width="100%">
</p>

## 📄 Abstract

Predicting spatial gene expression from pathology images depends on the image
representation, prediction head, and available spatial context. These factors
often change together, making performance differences difficult to attribute.
SimpleP2G separates them through controlled comparisons on **five HEST-1K
datasets and two 200-gene panels**.

**SimpleP2G** uses learnable gene queries to attend to frozen image embeddings.
**SimpleP2G-cross** adds spatially positioned neighboring embeddings. The
benchmark compares these models with eight baselines under explicit single-patch
and cross-patch settings.

## 📊 Results and Downloads

Each fold uses seed 42 and a separate validation slide for checkpoint selection. Encoder swaps pair a reference input with its Virchow2 counterpart on the same test slide.

| Download | Contents |
| --- | --- |
| [Main experiments · CSV](docs/results/main-results.csv) | 300 fits: 10 models × 5 datasets × 2 panels × 3 folds |
| [Image encoder swaps · CSV](docs/results/encoder-upgrades.csv) | 120 matched fold pairs: 4 methods × 5 datasets × 2 panels × 3 folds |

Both files contain unrounded PCC-200, PCC-50, MSE, and MAE, plus split IDs, selected epochs, and provenance hashes. See the [column definitions and aggregation example](docs/results/README.md).

With Virchow2 fixed, **SimpleP2G leads all 10 single-patch dataset–panel comparisons** in mean PCC-200. **SimpleP2G-cross leads 8 of 10**, with M2TGLGO ahead on PRAD HMHVG and ccRCC HVG.

| Dataset | SimpleP2G · HMHVG | SimpleP2G · HVG | SimpleP2G-cross · HMHVG | SimpleP2G-cross · HVG |
| --- | ---: | ---: | ---: | ---: |
| Brain | **0.4365 ± 0.1248** | **0.1722 ± 0.0620** | **0.4702 ± 0.1232** | **0.1964 ± 0.0463** |
| PRAD | **0.5138 ± 0.0914** | **0.2160 ± 0.0350** | 0.5541 ± 0.0974 | **0.2380 ± 0.0348** |
| HER2ST | **0.5078 ± 0.1998** | **0.2380 ± 0.1265** | **0.5423 ± 0.1906** | **0.2460 ± 0.1221** |
| Lung | **0.8592 ± 0.0127** | **0.4754 ± 0.0244** | **0.8791 ± 0.0100** | **0.4881 ± 0.0190** |
| ccRCC | **0.3508 ± 0.1458** | **0.1410 ± 0.0334** | **0.4490 ± 0.1023** | 0.1922 ± 0.0655 |

**PCC-200, mean ± sample SD across three test slides.** Bold marks the highest mean among all methods in the same input setting.

Upgrading four existing heads to Virchow2 improves **39 of 40** dataset–method–panel means. Paired mean changes range from **−0.0118 to +0.3113 PCC**; BLEEP on Brain HMHVG is the negative case. In the paper’s separate fold-0 comparison of six encoders with the same SimpleP2G head, Virchow2 leads on seven dataset–panel pairs, UNI on two, and DINOv2-L on one.

<a id="installation"></a>

## 🛠️ Installation

Use **Linux and Python 3.12**. The experiments used **PyTorch 2.10.0 with CUDA 12.8**. Exact package versions are listed in [requirements.txt](requirements.txt). GPU runs require a compatible NVIDIA driver.

Run these commands from the code repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
python -m pip install --no-deps -e .
simplep2g-train --help
```

<a id="pretrained-models"></a>

## 🤗 Pretrained Models

Download the public checkpoints from **[king159/SimpleP2G on Hugging Face](https://huggingface.co/king159/SimpleP2G)**.
The release contains **10 single-patch models**: five datasets × HMHVG-200/HVG-200,
using **fold 0 and seed 42**. Each model uses the EMA checkpoint selected by
validation PCC-200 after 100 training epochs. Each checkpoint includes its ordered
gene list, architecture, split IDs and evaluation metadata.

Each head takes a tensor of shape `(spots, 2560)`. The example below predicts expression for the first 16 spots on the Brain fold-0 test slide. First generate its embeddings using the [feature extraction instructions](docs/reproducibility.md#features-and-gene-panels).

```python
import torch
from src.pretrained import load_pretrained

model, genes, metadata = load_pretrained(
    "king159/SimpleP2G", dataset="brain", panel="hmhvg", device="cpu",
    revision="30e5a2403a420e793b0a8c9043901aeee01891f9",
)
embeddings = torch.load(
    "hest1k_datasets/brain/processed_data/1spot_virchow2_ebd/NCBI667_virchow2.pt",
    map_location="cpu", weights_only=True,
).float()
with torch.inference_mode():
    mean, log_variance = model(embeddings[:16])
# mean[:, i] predicts genes[i] in log2(raw_count + 1) units.
```

To download all ten heads for offline use, run

```bash
hf download king159/SimpleP2G --revision 30e5a2403a420e793b0a8c9043901aeee01891f9 --local-dir checkpoints/SimpleP2G
```

Pass `"checkpoints/SimpleP2G"` instead of the Hub ID to load from that directory.

## 🚀 Reproduce an experiment

Follow the [data and feature preparation guide](docs/reproducibility.md#data-and-weights) to download HEST and Virchow2, extract image embeddings and select genes from the training slides. For a first fit, prepare the Brain cohort and its fold-0 panels, then run from the repository root:

```bash
python scripts/build_benchmark.py --cohort brain --fold 0 --method simplep2g --output results/brain-single
simplep2g-train --config results/brain-single/configs/brain_fold0_hmhvg_simplep2g_seed42.yaml
```

This trains the HMHVG-200 single-patch model and saves `final_predictions.npz` and `run_summary.json`. To train SimpleP2G-cross with the target and its 24 nearest neighbors, select `--method simplep2g_cross` and a new output directory.

| Task | Guide |
| --- | --- |
| Run the ten methods across all splits and evaluate saved predictions | [Reproduce the benchmark](docs/reproducibility.md#training-and-checkpoint-selection) |
| Choose the method, training budget and slide assignments | [Experiment configurations](configs/README.md) |
| Reproduce the four paired image-encoder comparisons | [Reference inputs and caches](docs/encoders.md#reproduce-the-encoder-comparison) |
| Calculate PCC, error metrics and slide-level SD from the downloads | [Result columns and aggregation](docs/results/README.md) |

## 📄 License

Original SimpleP2G code uses the [MIT license](LICENSE). See the [upstream notices](THIRD_PARTY_NOTICES.md) for baseline implementations. The released prediction heads have an [MIT-licensed model card](https://huggingface.co/king159/SimpleP2G). Data and encoder weights are obtained under their publishers’ terms.
