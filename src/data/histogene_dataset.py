"""Slide-level dataset for HisToGene.

Original HisToGene batches by **slide**, not by spot.  Each ``__getitem__``
returns *all* spots of a single slide so the Transformer's self-attention can
model inter-spot spatial relationships (cross-spot attention).

Two modes:
  * **enhanced** – uses precomputed foundation-model embeddings loaded from
    ``.pt`` files (same pipeline as SimpleP2G/STeM).
  * **original** – loads explicitly prepared raw-patch caches and flattens
    them, matching the original HisToGene implementation.
"""

from __future__ import annotations

import os

import anndata
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from src.data.hest_wsi_dataset import WSIPatchConfig, _imagenet_normalize


class HisToGeneSlideDataset(Dataset):
    """Each item = one slide with all its spots.

    Returns
    -------
    dict with keys:
        labels     : (N_spots, G) float32 – log2(count+1) normalised expression
        pos_bins   : (N_spots, 2) long    – discretised (x, y) grid indices
        embeddings : (N_spots, D) float32 – foundation-model embeddings  (enhanced only)
        patches    : (N_spots, patch_dim) float32 – flattened raw patches   (original only)
    """

    def __init__(
        self,
        slide_ids: list[str],
        data_path: str,
        selected_genes: list[str],
        *,
        # Mode
        use_embeddings: bool = True,
        # Embedding-mode settings
        encoder_name: str = "virchow2",
        ebd_dir: str | None = None,
        ebd_suffix: str | None = None,
        # Raw-patch-mode settings
        patch_cfg: WSIPatchConfig | None = None,
        raw_patch_cache_dir: str | None = None,
        # Shared
        n_pos_bins: int = 64,
        max_spots: int | None = None,  # random subsample cap (training)
        train: bool = True,
    ):
        super().__init__()
        self.slide_ids = list(slide_ids)
        self.data_path = data_path
        self.selected_genes = selected_genes
        self.use_embeddings = use_embeddings
        self.encoder_name = encoder_name
        self.n_pos_bins = n_pos_bins
        self.max_spots = max_spots
        self.train = train
        self.patch_cfg = patch_cfg or WSIPatchConfig()
        if raw_patch_cache_dir is None:
            raw_patch_cache_dir = os.path.join(
                data_path,
                "processed_data",
                (f"histogene_raw_patch_cache_ps{self.patch_cfg.patch_size}_level{self.patch_cfg.level}"),
            )
        elif not os.path.isabs(raw_patch_cache_dir):
            raw_patch_cache_dir = os.path.join(data_path, raw_patch_cache_dir)
        self.raw_patch_cache_dir = raw_patch_cache_dir
        self._raw_patch_arrays: dict[str, np.ndarray] = {}

        # Resolve embedding directory / suffix
        if use_embeddings:
            from src.training_support import ENCODER_REGISTRY

            if ebd_dir is None:
                _, ebd_dir_name, _, suffix = ENCODER_REGISTRY[encoder_name]
                self.ebd_dir = os.path.join(data_path, "processed_data", ebd_dir_name)
                self.ebd_suffix = suffix
            else:
                self.ebd_dir = os.path.join(data_path, "processed_data", ebd_dir)
                self.ebd_suffix = ebd_suffix or encoder_name

        # Pre-load metadata per slide (adata + spatial coords)
        self.slide_data: list[dict] = []
        for sid in self.slide_ids:
            h5ad_path = os.path.join(data_path, "st", f"{sid}.h5ad")
            adata = anndata.read_h5ad(h5ad_path)
            adata_sel = adata[:, selected_genes]
            X = adata_sel.X
            if sparse.issparse(X):
                X = X.toarray()
            labels = np.log2(X.astype(np.float32) + 1.0)

            # Spatial coordinates
            if "spatial" in adata.obsm:
                coords = adata.obsm["spatial"].astype(np.float64)
            else:
                raise ValueError(f"Missing spatial coordinates for {sid}")

            self.slide_data.append(
                {
                    "slide_id": sid,
                    "labels": labels,
                    "coords": coords,
                }
            )

    def __len__(self) -> int:
        return len(self.slide_ids)

    def _discretize_coords(self, coords: np.ndarray) -> np.ndarray:
        """Convert raw spatial (x, y) to integer grid bins in [0, n_pos_bins)."""
        out = np.zeros_like(coords, dtype=np.int64)
        for dim in range(2):
            c = coords[:, dim]
            mn, mx = c.min(), c.max()
            if mx - mn > 0:
                normed = (c - mn) / (mx - mn)
            else:
                normed = np.zeros_like(c)
            out[:, dim] = np.clip((normed * (self.n_pos_bins - 1)).round(), 0, self.n_pos_bins - 1).astype(np.int64)
        return out

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        info = self.slide_data[index]
        sid = info["slide_id"]
        labels = info["labels"]  # (N, G)
        coords = info["coords"]  # (N, 2)
        n_spots = labels.shape[0]

        # Optional subsampling for training
        if self.max_spots is not None and self.train and n_spots > self.max_spots:
            idx = np.random.choice(n_spots, self.max_spots, replace=False)
            idx.sort()
        else:
            idx = np.arange(n_spots)

        labels_sel = labels[idx]
        coords_sel = coords[idx]
        pos_bins = self._discretize_coords(coords_sel)

        result: dict[str, torch.Tensor] = {
            "labels": torch.from_numpy(labels_sel).float(),
            "pos_bins": torch.from_numpy(pos_bins).long(),
        }

        if self.use_embeddings:
            # Load precomputed embeddings
            pt_path = os.path.join(self.ebd_dir, f"{sid}_{self.ebd_suffix}.pt")
            ebd = torch.load(pt_path, map_location="cpu")  # (N_all, D)
            result["embeddings"] = ebd[idx].float()
        else:
            ps = self.patch_cfg.patch_size
            cache_path = os.path.join(self.raw_patch_cache_dir, f"{sid}.npy")
            if os.path.exists(cache_path):
                cached = self._raw_patch_arrays.get(sid)
                if cached is None:
                    cached = np.load(cache_path, mmap_mode="r")
                    expected_shape = (n_spots, 3, ps, ps)
                    if cached.dtype != np.uint8 or cached.shape != expected_shape:
                        raise ValueError(
                            f"Invalid HisToGene patch cache {cache_path}: "
                            f"expected uint8 {expected_shape}, got "
                            f"{cached.dtype} {cached.shape}"
                        )
                    self._raw_patch_arrays[sid] = cached
                # Advanced indexing materializes only the requested slide/spot
                # rows; the underlying deterministic RGB cache remains mmap'ed.
                patch_tensor = torch.from_numpy(np.asarray(cached[idx])).float()
                patch_tensor.div_(255.0)
                if self.patch_cfg.imagenet_norm:
                    patch_tensor = _imagenet_normalize(patch_tensor)
                result["patches"] = patch_tensor.flatten(1)
            else:
                raise FileNotFoundError(
                    f"Required raw-patch cache is missing: {cache_path}. "
                    "Run scripts/cache_histogene_raw_patches.py first."
                )

        return result
