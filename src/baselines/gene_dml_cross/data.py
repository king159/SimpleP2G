"""Strict, image-only slide cache interface for the Gene-DML cross prototype.

No conversion from pooled embeddings or nearest measured spots is provided.
Token caches use memory-mapped NPY files because full target sequences are
substantially larger than ordinary per-spot pooled features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .model import UPSTREAM_COMMIT


SCHEMA = "gene_dml_cross_image_cache_v1"
INPUT_KEYS = ("target_tokens", "neighbor_features", "neighbor_mask", "global_features", "positions", "spot_indices")


def restore_mixed_spot_order(batch, expression, grouped_row_indices):
    """Undo I/O grouping before KMeans uses the first K batch rows as centers."""
    original_rows = np.asarray(grouped_row_indices, dtype=np.int64)
    size = len(expression)
    if original_rows.shape != (size,) or not np.array_equal(np.sort(original_rows), np.arange(size)):
        raise ValueError("Grouped row indices must be a permutation of the original mini-batch rows")
    order = torch.tensor(np.argsort(original_rows), dtype=torch.long, device=expression.device)
    result = dict(batch)
    for key in ("target_tokens", "neighbor_features", "neighbor_mask", "spot_indices", "slide_indices"):
        result[key] = batch[key].index_select(0, order.to(batch[key].device))
    return result, expression.index_select(0, order)


def mixed_slide_batch(caches, indices_per_slide, device="cpu"):
    """Preserve mixed-slide contrastive batches while deduplicating slide images.

    Each cache appears once. Caller-selected spot indices define the sampling
    protocol; this helper does not replace pooled-spot sampling by slide-balanced
    sampling or change the batch's loss negatives.
    """
    if not caches or len(caches) != len(indices_per_slide):
        raise ValueError("Provide one index list per slide cache")
    if len({cache.slide_id for cache in caches}) != len(caches):
        raise ValueError("Provide each slide cache once, with all of its selected spots")
    batches = [cache.batch(indices, device) for cache, indices in zip(caches, indices_per_slide)]
    result = {
        key: torch.cat([batch[key] for batch in batches])
        for key in ("target_tokens", "neighbor_features", "neighbor_mask", "spot_indices")
    }
    result["global_features"] = [batch["global_features"] for batch in batches]
    result["positions"] = [batch["positions"] for batch in batches]
    result["slide_indices"] = torch.cat(
        [
            torch.full((len(batch["spot_indices"]),), index, dtype=torch.long, device=device)
            for index, batch in enumerate(batches)
        ]
    )
    return result


def image_grid_boxes(center_xy, patch_size=224):
    """Actual 5x5 contiguous image tiles, centered on the target, in row order.

    Returns (x_left, y_top, x_right, y_bottom), in level-zero image pixels.
    The native released NumPy preprocessing uses a 224-pixel target footprint.
    An alternative physical patch footprint must be recorded as an adaptation.
    """
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("patch_size must be a positive even integer")
    center = np.asarray(center_xy)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy must be a finite pair of full-resolution image coordinates")
    center = np.floor(center).astype(np.int64)
    start = center - (5 * patch_size) // 2
    boxes = []
    for row in range(5):
        for col in range(5):
            x, y = start + np.array([col, row]) * patch_size
            boxes.append((x, y, x + patch_size, y + patch_size))
    return np.asarray(boxes, dtype=np.int64)


def image_grid_mask(center_xy, image_wh, patch_size=224, *, allow_target_padding=False):
    """Native NumPy branch's 5x5 border mask, with opt-in padded targets.

    For a valid target, the upstream window=5/3 mask is exactly the full-tile
    boundary test below. The center tile is included (index 12).
    """
    boxes = image_grid_boxes(center_xy, patch_size)
    width, height = image_wh
    valid = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= width) & (boxes[:, 3] <= height)
    if not valid[12] and allow_target_padding:
        # Only the target may be partially padded by OpenSlide. Other grid
        # cells still follow the native fully-inside tile mask. A completely
        # out-of-image target remains a black RGB patch, as in the benchmark
        # image reader; its spot and coordinates are not discarded or moved.
        valid[12] = True
    if not valid[12]:
        raise ValueError(
            "Target crosses the WSI boundary; define and record a target-padding policy before extraction"
        )
    return valid


def split_backbone_tokens(tokens):
    """Keep all target tokens and use CLS for native-width context descriptors.

    UNI uses a 1024-d CLS descriptor. Virchow2's analogous CLS descriptor is
    1280-d; its four register tokens remain in the full target sequence. This
    is explicitly different from the benchmark's cached 2560-d concatenation.
    """
    if tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError("A backbone token sequence is required; a pooled vector cannot reconstruct it")
    return tokens, tokens[:, 0]


class GeneDMLSlideCache:
    """Load one slide's image-only cache, enforcing spot identity and provenance.

    JSON has ``schema``, ``upstream_commit``, ``slide_id``, ``encoder``,
    ``encoder_revision``, ``preprocessing``, ``neighbor_layout='image_grid_5x5'``,
    ``target_representation='full_sequence'``, ``context_representation='cls'``,
    ``position_space='st_lattice'``, and ``arrays`` mapping names to NPY paths.
    Arrays: target_tokens, neighbor_features, neighbor_mask, global_features,
    positions, spot_ids. No expression matrix is stored in this image cache.
    """

    def __init__(self, manifest_path):
        path = Path(manifest_path)
        self.metadata = json.loads(path.read_text())
        required = {
            "schema": SCHEMA,
            "upstream_commit": UPSTREAM_COMMIT,
            "neighbor_layout": "image_grid_5x5",
            "target_representation": "full_sequence",
            "context_representation": "cls",
            "position_space": "st_lattice",
        }
        for key, expected in required.items():
            if self.metadata.get(key) != expected:
                raise ValueError(f"Cache {key} must be {expected!r}; got {self.metadata.get(key)!r}")
        for key in ("slide_id", "encoder", "encoder_revision", "preprocessing"):
            if not self.metadata.get(key):
                raise ValueError(f"Missing feature provenance: {key}")
        self.arrays = {}
        for key in ("target_tokens", "neighbor_features", "neighbor_mask", "global_features", "positions", "spot_ids"):
            array_path = Path(self.metadata["arrays"][key])
            self.arrays[key] = np.load(
                array_path if array_path.is_absolute() else path.parent / array_path, mmap_mode="r", allow_pickle=False
            )
        if "target_tokens_aug" in self.metadata["arrays"]:
            aug_path = Path(self.metadata["arrays"]["target_tokens_aug"])
            self.arrays["target_tokens_aug"] = np.load(
                aug_path if aug_path.is_absolute() else path.parent / aug_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        target = self.arrays["target_tokens"]
        if target.ndim != 3 or target.shape[1] < 2:
            raise ValueError("Cache must contain full target token sequences")
        n, _, width = target.shape
        if "target_tokens_aug" in self.arrays:
            augmented = self.arrays["target_tokens_aug"]
            if augmented.ndim != 4 or augmented.shape[0] != n or augmented.shape[2:] != target.shape[1:]:
                raise ValueError("Augmented target cache must have shape (N, views, T, D)")
            if len(self.metadata.get("target_augmentation_names", [])) != augmented.shape[1]:
                raise ValueError("Augmented target views need explicit names in cache provenance")
        expected_shapes = {
            "neighbor_features": (n, 25, width),
            "neighbor_mask": (n, 25),
            "global_features": (n, width),
            "positions": (n, 2),
            "spot_ids": (n,),
        }
        for key, shape in expected_shapes.items():
            if self.arrays[key].shape != shape:
                raise ValueError(f"Cache {key} has {self.arrays[key].shape}, expected {shape}")
        if n == 0:
            raise ValueError("An empty slide cache is invalid")
        if self.arrays["positions"].dtype.kind not in "iu":
            raise ValueError("Global positions must be integer ST lattice coordinates")
        if len(np.unique(self.arrays["spot_ids"])) != n:
            raise ValueError("Cache spot IDs must be unique")
        if len(np.unique(self.arrays["positions"], axis=0)) != n:
            raise ValueError("Cache lattice positions must be unique")
        self.slide_id = self.metadata["slide_id"]

    def __len__(self):
        return len(self.arrays["target_tokens"])

    def batch(self, indices, device="cpu", view_indices=None, *, include_global=True):
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or indices.min() < 0 or indices.max() >= len(self):
            raise ValueError("Batch indices must be nonempty, one-dimensional, and within this slide")
        batch = {}
        for key in ("target_tokens", "neighbor_features", "neighbor_mask"):
            array = np.array(self.arrays[key][indices], copy=True)
            if key == "target_tokens" and view_indices is not None:
                view_indices = np.asarray(view_indices, dtype=np.int64)
                augmented = self.arrays.get("target_tokens_aug")
                if augmented is None:
                    raise ValueError("Requested target augmentation but the cache has no augmented token views")
                if (
                    view_indices.shape != indices.shape
                    or view_indices.min() < 0
                    or view_indices.max() > augmented.shape[1]
                ):
                    raise ValueError("View indices must align with spots: 0=identity, 1..A=cached augmentations")
                selected = view_indices > 0
                array[selected] = augmented[indices[selected], view_indices[selected] - 1]
            batch[key] = torch.from_numpy(array).to(
                device=device, dtype=torch.bool if key == "neighbor_mask" else torch.float32
            )
        # Training runners may already keep these immutable whole-slide inputs
        # on the GPU. Avoid copying the complete arrays for every mini-batch.
        for key in (("global_features", "positions") if include_global else ()):
            batch[key] = torch.from_numpy(np.array(self.arrays[key], copy=True)).to(
                device=device,
                dtype=torch.long if key == "positions" else torch.float32,
            )
        batch["spot_indices"] = torch.tensor(indices, dtype=torch.long, device=device)
        return batch

    def align_expression(self, values, spot_ids, indices, device="cpu"):
        """Join training/evaluation targets by explicit ID; never assume row order."""
        values = np.asarray(values)
        spot_ids = np.asarray(spot_ids).astype(str)
        if values.ndim != 2 or len(values) != len(spot_ids) or len(np.unique(spot_ids)) != len(spot_ids):
            raise ValueError("Expression must have unique, row-aligned spot IDs")
        lookup = {spot: i for i, spot in enumerate(spot_ids)}
        requested = self.arrays["spot_ids"][indices].astype(str)
        try:
            selected = [lookup[spot] for spot in requested]
        except KeyError as error:
            raise ValueError(f"Expression is missing cache spot {error.args[0]}") from error
        return torch.tensor(values[selected], dtype=torch.float32, device=device)
