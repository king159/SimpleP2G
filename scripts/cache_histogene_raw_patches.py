#!/usr/bin/env python3
"""Cache deterministic HisToGene RGB spot patches as memory-mapped uint8 arrays.

The released HisToGene input is a flattened 112x112 RGB patch. Re-reading and
decoding every WSI patch for every epoch dominates runtime, although the pixels
never change. This script performs that deterministic extraction once. The
dataset applies the original float conversion and ImageNet normalization after
loading, so model inputs and the training protocol are unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import anndata
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hest_wsi_dataset import OpenSlideReader  # noqa: E402


DEFAULT_DATASETS = ("brain", "prad", "her2st", "lung", "ccrcc")


def _slide_ids(dataset_dir: Path) -> list[str]:
    return sorted(p.stem for p in (dataset_dir / "st").glob("*.h5ad"))


def _valid_cache(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        cached = np.load(path, mmap_mode="r")
        return cached.dtype == np.uint8 and cached.shape == shape
    except Exception:
        return False


def _cache_slide(
    dataset_dir: Path,
    slide_id: str,
    patch_size: int,
    level: int,
) -> tuple[str, int, float, bool]:
    adata = anndata.read_h5ad(dataset_dir / "st" / f"{slide_id}.h5ad", backed="r")
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    if getattr(adata, "file", None) is not None:
        adata.file.close()

    cache_dir = dataset_dir / "processed_data" / f"histogene_raw_patch_cache_ps{patch_size}_level{level}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{slide_id}.npy"
    shape = (len(coords), 3, patch_size, patch_size)
    if _valid_cache(output_path, shape):
        return slide_id, len(coords), 0.0, True

    wsi_path = dataset_dir / "wsis" / f"{slide_id}.tif"
    if not wsi_path.exists():
        raise FileNotFoundError(f"Missing WSI for {slide_id}: {wsi_path}")

    started = time.perf_counter()
    temporary_path = cache_dir / f".{slide_id}.{os.getpid()}.tmp.npy"
    cache = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.uint8,
        shape=shape,
    )
    reader = OpenSlideReader(str(wsi_path))
    half = patch_size // 2
    for index, (cx, cy) in enumerate(coords):
        image = reader.read_region_rgb(
            int(cx - half),
            int(cy - half),
            patch_size,
            level,
        )
        cache[index] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
    cache.flush()
    del cache
    os.replace(temporary_path, output_path)
    return slide_id, len(coords), time.perf_counter() - started, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "hest1k_datasets")
    parser.add_argument("--patch-size", type=int, default=112)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    jobs: list[tuple[Path, str]] = []
    for dataset_name in args.datasets:
        dataset_dir = args.data_root / dataset_name
        slide_ids = _slide_ids(dataset_dir)
        if not slide_ids:
            parser.error(f"No .h5ad slides under {dataset_dir / 'st'}")
        jobs.extend((dataset_dir, slide_id) for slide_id in slide_ids)
    jobs = jobs[args.shard_index :: args.num_shards]

    started = time.perf_counter()
    print(
        f"Shard {args.shard_index}/{args.num_shards}: {len(jobs)} slide(s)",
        flush=True,
    )
    for job_index, (dataset_dir, slide_id) in enumerate(jobs, start=1):
        sid, spots, seconds, skipped = _cache_slide(
            dataset_dir,
            slide_id,
            args.patch_size,
            args.level,
        )
        status = "cached" if skipped else f"{seconds:.1f}s"
        print(
            f"[{job_index}/{len(jobs)}] {dataset_dir.name}/{sid}: {spots} spots ({status})",
            flush=True,
        )
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
