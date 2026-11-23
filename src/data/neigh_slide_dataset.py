"""Slide chunks with target and neighbor context from one frozen encoder.

Neighbor indices and relative positions are cached; feature tensors are gathered
lazily from the same per-slide target embeddings.
"""

from __future__ import annotations

import os

import anndata
import numpy as np
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


def build_spatial_neighbors(
    coords: np.ndarray,
    num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build nearest-neighbour indices and scale-normalised relative positions.

    The target spot itself is excluded.  When a slide contains fewer than
    ``num_neighbors + 1`` spots, missing entries point to the target but are
    masked out.  Relative positions are divided by the median nearest-neighbour
    spacing so that their scale is comparable across sequencing platforms.
    """

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape (N, 2), got {coords.shape}")
    if num_neighbors < 0:
        raise ValueError(f"num_neighbors must be non-negative, got {num_neighbors}")

    n_spots = coords.shape[0]
    indices = np.zeros((n_spots, num_neighbors), dtype=np.int64)
    mask = np.zeros((n_spots, num_neighbors), dtype=np.float32)
    relative = np.zeros((n_spots, num_neighbors, 2), dtype=np.float32)
    if n_spots == 0 or num_neighbors == 0:
        return (
            torch.from_numpy(indices),
            torch.from_numpy(mask),
            torch.from_numpy(relative),
        )

    finite_rows = np.isfinite(coords).all(axis=1)
    safe_coords = coords.copy()
    if not finite_rows.all():
        # Keep invalid-coordinate spots usable as targets while preventing the
        # KD-tree from failing. Their context remains fully masked.
        safe_coords[~finite_rows] = 0.0

    tree = cKDTree(safe_coords[finite_rows])
    valid_global = np.flatnonzero(finite_rows)
    query_k = min(num_neighbors + 1, max(valid_global.size, 1))
    nearest_spacing: list[float] = []

    for target_idx in range(n_spots):
        indices[target_idx, :] = target_idx
        if not finite_rows[target_idx] or valid_global.size <= 1:
            continue

        distances, local_indices = tree.query(safe_coords[target_idx], k=query_k)
        distances = np.atleast_1d(distances)
        local_indices = np.atleast_1d(local_indices)
        candidates: list[tuple[int, float]] = []
        for distance, local_idx in zip(distances, local_indices):
            neighbor_idx = int(valid_global[int(local_idx)])
            if neighbor_idx == target_idx or not np.isfinite(distance):
                continue
            candidates.append((neighbor_idx, float(distance)))
            if len(candidates) == num_neighbors:
                break

        for slot, (neighbor_idx, distance) in enumerate(candidates):
            indices[target_idx, slot] = neighbor_idx
            mask[target_idx, slot] = 1.0
            relative[target_idx, slot] = safe_coords[neighbor_idx] - safe_coords[target_idx]
            if slot == 0 and distance > 0:
                nearest_spacing.append(distance)

    spacing = float(np.median(nearest_spacing)) if nearest_spacing else 1.0
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0
    relative = np.clip(relative / spacing, -16.0, 16.0)
    relative *= mask[..., None]
    return (
        torch.from_numpy(indices),
        torch.from_numpy(mask),
        torch.from_numpy(relative),
    )


class NeighSlideDataset(Dataset):
    """One item = one slide chunk with target and neighboring embeddings."""

    def __init__(
        self,
        slide_ids: list[str],
        data_path: str,
        selected_genes: list[str],
        ebd_dir: str,
        ebd_suffix: str,
        aug_dir: str | None,
        aug_suffix: str | None,
        neighbor_dim: int,
        num_neighbor_tokens: int,
        train: bool = True,
        max_spots: int | None = None,
        filter_bad_spots: bool = False,
        chunks_per_slide: int = 1,
        num_aug_ratio: int = 0,
        sample_augmented_targets: bool = False,
    ):
        super().__init__()
        self.slide_ids = list(slide_ids)
        self.train = train
        self.max_spots = max_spots
        self.filter_bad_spots = filter_bad_spots
        self.chunks_per_slide = max(1, int(chunks_per_slide))
        self.num_aug_ratio = max(0, int(num_aug_ratio))
        self.sample_augmented_targets = bool(sample_augmented_targets)
        self.cache: list[dict] = []
        processed_root = os.path.join(data_path, "processed_data")
        ebd_root = os.path.join(processed_root, ebd_dir)
        aug_root = os.path.join(processed_root, aug_dir) if aug_dir else None
        for sid in self.slide_ids:
            adata = anndata.read_h5ad(os.path.join(data_path, "st", f"{sid}.h5ad"))
            X = adata[:, selected_genes].X
            if sparse.issparse(X):
                X = X.toarray()
            labels_np = X.astype(np.float32)

            embeddings = torch.load(os.path.join(ebd_root, f"{sid}_{ebd_suffix}.pt"), map_location="cpu").float()
            embeddings_aug = None
            if self.train and self.sample_augmented_targets and self.num_aug_ratio > 0:
                if not aug_root or not aug_suffix:
                    raise ValueError("Augmented training requires an explicit augmentation cache")
                aug_path = os.path.join(aug_root, f"{sid}_{aug_suffix}_aug.pt")
                embeddings_aug = torch.load(aug_path, map_location="cpu").float()
            n_spots = embeddings.shape[0]
            if "spatial" not in adata.obsm:
                raise ValueError(f"Missing spatial coordinates for {sid}")
            coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

            labels = torch.from_numpy(np.log2(labels_np + 1.0)).float()
            keep = torch.ones(n_spots, dtype=torch.bool)
            if filter_bad_spots:
                raw = torch.from_numpy(labels_np)
                keep &= ~torch.isnan(raw).all(dim=1)
                keep &= raw.nan_to_num().sum(dim=1) != 0

            embeddings = embeddings[keep]
            embeddings_aug = embeddings_aug[keep] if embeddings_aug is not None else None
            labels = labels[keep]
            coords = coords[keep.numpy()]
            n_spots = embeddings.shape[0]

            if neighbor_dim != embeddings.shape[-1]:
                raise ValueError(
                    "Unified context must use the target encoder dimension: "
                    f"neighbor_dim={neighbor_dim}, target_dim={embeddings.shape[-1]}"
                )
            neighbor_indices, neighbor_mask, neighbor_pos = build_spatial_neighbors(
                coords, num_neighbor_tokens,
            )
            self.cache.append(
                {
                    "slide_id": sid,
                    "embeddings": embeddings,
                    "embeddings_aug": embeddings_aug,
                    "labels": labels,
                    "neighbor_indices": neighbor_indices,
                    "neighbor_mask": neighbor_mask.float(),
                    "neighbor_pos": neighbor_pos.float(),
                }
            )

    def __len__(self) -> int:
        if self.train:
            return len(self.slide_ids) * self.chunks_per_slide
        return len(self.slide_ids)

    def __getitem__(self, index: int) -> dict:
        info = self.cache[index % len(self.cache)]
        n_spots = info["embeddings"].shape[0]
        if self.train and self.max_spots is not None and n_spots > self.max_spots:
            idx = np.random.choice(n_spots, self.max_spots, replace=False)
            idx.sort()
            idx = torch.from_numpy(idx).long()
        else:
            idx = torch.arange(n_spots)

        embeddings = info["embeddings"][idx]
        embeddings_aug = info.get("embeddings_aug")
        if self.train and embeddings_aug is not None and self.sample_augmented_targets and self.num_aug_ratio > 0:
            aug = embeddings_aug[idx]
            aug_idx = torch.randint(0, aug.shape[1], (idx.numel(),), dtype=torch.long)
            use_aug = torch.rand(idx.numel()) < (self.num_aug_ratio / (self.num_aug_ratio + 1.0))
            aug_view = aug[torch.arange(idx.numel()), aug_idx]
            embeddings = torch.where(use_aug[:, None], aug_view, embeddings)

        neighbor_indices = info["neighbor_indices"][idx]
        neighbor_emb = info["embeddings"][neighbor_indices]

        return {
            "slide_id": info["slide_id"],
            "embeddings": embeddings,
            "labels": info["labels"][idx],
            "neighbor_emb": neighbor_emb,
            "neighbor_mask": info["neighbor_mask"][idx],
            "neighbor_pos": info["neighbor_pos"][idx],
        }
