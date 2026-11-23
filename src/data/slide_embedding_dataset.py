"""Slide-level dataset for cross-patch baselines (STFlow, MERGE).

Each ``__getitem__`` returns one slide: precomputed spot embeddings, raw
spatial coordinates, log2(count+1) gene expression for ``selected_genes``.

Embeddings are taken from ``processed_data/1spot_<encoder>_ebd/`` (matches
SimpleP2G's data layout).
"""

from __future__ import annotations

import os

import anndata
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class SlideEmbeddingDataset(Dataset):
    """One item = one slide with all its spots.

    Returns dict:
        embeddings : (N, D) float32
        coords     : (N, 2) float32 (raw pixel coords, normalised per-slide)
        labels     : (N, G) float32 (log2(count+1))
        slide_id   : str
    """

    def __init__(
        self,
        slide_ids: list[str],
        data_path: str,
        selected_genes: list[str],
        encoder_name: str,
        ebd_dir: str,
        ebd_suffix: str,
        train: bool = True,
        max_spots: int | None = None,
        coord_scale: str = "minmax",  # "minmax" | "raw"
    ):
        super().__init__()
        self.slide_ids = list(slide_ids)
        self.data_path = data_path
        self.selected_genes = selected_genes
        self.encoder_name = encoder_name
        self.ebd_dir = os.path.join(data_path, "processed_data", ebd_dir)
        self.ebd_suffix = ebd_suffix
        self.train = train
        self.max_spots = max_spots
        self.coord_scale = coord_scale

        # Cache per-slide tensors
        self.cache: list[dict] = []
        for sid in self.slide_ids:
            adata = anndata.read_h5ad(os.path.join(data_path, "st", f"{sid}.h5ad"))
            X = adata[:, selected_genes].X
            if sparse.issparse(X):
                X = X.toarray()
            labels = np.log2(X.astype(np.float32) + 1.0)

            ebd_path = os.path.join(self.ebd_dir, f"{sid}_{self.ebd_suffix}.pt")
            embeddings = torch.load(ebd_path, map_location="cpu").float()  # (N, D)

            if "spatial" not in adata.obsm:
                raise ValueError(f"Missing spatial coordinates for {sid}")
            coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
            if coord_scale == "minmax":
                mn, mx = coords.min(axis=0, keepdims=True), coords.max(axis=0, keepdims=True)
                rng = np.where(mx - mn > 0, mx - mn, 1.0)
                coords = (coords - mn) / rng  # in [0, 1]
            self.cache.append(
                {
                    "slide_id": sid,
                    "embeddings": embeddings,
                    "coords": torch.from_numpy(coords).float(),
                    "labels": torch.from_numpy(labels).float(),
                }
            )

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        info = self.cache[index]
        embeddings = info["embeddings"]
        coords = info["coords"]
        labels = info["labels"]
        n_spots = embeddings.shape[0]
        if self.max_spots is not None and self.train and n_spots > self.max_spots:
            idx = np.random.choice(n_spots, self.max_spots, replace=False)
            idx.sort()
            embeddings = embeddings[idx]
            coords = coords[idx]
            labels = labels[idx]
        return {
            "embeddings": embeddings,
            "coords": coords,
            "labels": labels,
            "slide_id": info["slide_id"],
        }
