"""Load released SimpleP2G prediction heads without an image encoder."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from src.simpleP2G import SimpleP2G


DATASETS = ("brain", "prad", "her2st", "lung", "ccrcc")
PANELS = ("hmhvg", "hvg")
SETTINGS = ("single-patch", "cross-patch")


def load_pretrained(
    repo_id_or_path="king159/SimpleP2G",
    *,
    dataset="brain",
    panel="hmhvg",
    setting="single-patch",
    device="cpu",
    revision=None,
    cache_dir=None,
    local_files_only=False,
):
    """Return an eval-mode model, ordered gene names and checkpoint metadata.

    ``repo_id_or_path`` is a Hub model ID or a downloaded repository directory.
    Inputs are precomputed image embeddings; no image-encoder weights are loaded.
    Predictions follow the order in ``genes`` and use log2(count + 1) units.
    """
    dataset, panel = dataset.lower(), panel.lower()
    if dataset not in DATASETS or panel not in PANELS or setting not in SETTINGS:
        raise ValueError(f"Unsupported model selection: {dataset}/{panel}/{setting}")
    folder = f"models/{setting}/{dataset}/{panel}"
    local = Path(repo_id_or_path).expanduser()

    def resolve(filename):
        if local.is_dir():
            path = local / folder / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            repo_id=str(repo_id_or_path), filename=f"{folder}/{filename}",
            revision=revision, cache_dir=cache_dir, local_files_only=local_files_only,
        ))

    metadata = json.loads(resolve("config.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint metadata version")
    expected = {"dataset": dataset, "panel": panel, "setting": setting}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint metadata does not match the requested model")
    genes = json.loads(resolve("genes.json").read_text())
    config = SimpleNamespace(**metadata["model_config"])
    if len(genes) != config.num_genes or len(set(genes)) != len(genes):
        raise ValueError("The ordered gene list must contain one unique name per output")
    if setting == "single-patch":
        model = SimpleP2G(config)
    else:
        from src.simpleP2G_neigh import SimpleP2GNeigh

        model = SimpleP2GNeigh(config)
    state = load_file(str(resolve("model.safetensors")), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    return model, genes, metadata
