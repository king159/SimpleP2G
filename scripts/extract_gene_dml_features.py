"""Extract full Virchow2 target sequences, 5x5 neighbors and slide CLS tokens."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ENCODER_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()

def extract(args, plan):
    import anndata
    import openslide
    import timm
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_data_config
    from timm.layers import SwiGLUPacked

    from src.baselines.gene_dml_cross.data import SCHEMA, GeneDMLSlideCache, image_grid_boxes, image_grid_mask

    if not torch.cuda.is_available():
        raise RuntimeError("Extraction must execute on a CUDA GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_dir = Path(plan["encoder_directory"])
    model_config = json.loads((model_dir / "config.json").read_text())
    model = timm.create_model(
        model_config["architecture"],
        pretrained=False,
        pretrained_cfg=model_config["pretrained_cfg"],
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        **model_config["model_args"],
    )
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model = model.cuda().eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model_hash = digest(model_dir / "model.safetensors")
    selected = plan["slides"][args.shard :: plan["extraction_shards"]]
    if args.slide:
        selected = [r for r in plan["slides"] if r["slide"] == args.slide]
        if not selected:
            raise ValueError("Requested slide is outside the requested slide list")

    @torch.inference_mode()
    def encode(batch):
        with torch.autocast("cuda", dtype=torch.float16):
            tokens = model(batch.cuda(non_blocking=True))
        if tokens.shape[1:] != (261, 1280) or not torch.isfinite(tokens).all():
            raise ValueError("Unexpected or non-finite Virchow2 tokens")
        return tokens.half().cpu().numpy()

    zero_normalized_cls = encode(torch.zeros(1, 3, 224, 224))[0, 0]
    cache_root, data = Path(plan["cache_root"]), Path(plan["data_root"])
    for record in selected:
        slide_id = record["slide"]
        output = cache_root / slide_id
        marker = output / "manifest.json"
        if marker.exists():
            cache = GeneDMLSlideCache(marker)
            if cache.metadata["encoder_revision"] != ENCODER_REVISION or cache.metadata["model_sha256"] != model_hash:
                raise RuntimeError("Existing cache provenance differs")
            print(f"CACHE ALREADY COMPLETE {slide_id}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()
        a = anndata.read_h5ad(data / "st" / f"{slide_id}.h5ad", backed="r")
        coords = np.asarray(a.obsm["spatial"])
        positions = a.obs[["array_row", "array_col"]].to_numpy(dtype=np.int64)
        spot_ids = np.asarray(a.obs_names, dtype=str)
        assert len(coords) == record["spots"]
        a.file.close()
        n = len(coords)
        slide = openslide.OpenSlide(str(data / "wsis" / f"{slide_id}.tif"))
        masks = np.stack([
            image_grid_mask(center, slide.dimensions, allow_target_padding=plan["allow_target_padding"])
            for center in coords
        ])
        arrays = {}
        for name, shape in (
            ("target_tokens", (n, 261, 1280)),
            ("target_tokens_aug", (n, 2, 261, 1280)),
            ("neighbor_features", (n, 25, 1280)),
            ("global_features", (n, 1280)),
        ):
            arrays[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float16, shape=shape)
        np.save(output / "positions.npy", positions)
        np.save(output / "spot_ids.npy", spot_ids)
        np.save(output / "neighbor_mask.npy", masks)
        target_started = time.perf_counter()
        batch_size = plan["extraction_batch_size"]
        for start in range(0, n, batch_size):
            patches = []
            for center in coords[start : start + batch_size]:
                x, y, _, _ = image_grid_boxes(center)[12]
                patches.append(slide.read_region((int(x), int(y)), 0, (224, 224)).convert("RGB"))
            for view, operation in enumerate((None, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM)):
                images = patches if operation is None else [image.transpose(operation) for image in patches]
                tokens = encode(torch.stack([transform(image) for image in images]))
                if view == 0:
                    arrays["target_tokens"][start : start + len(patches)] = tokens
                    arrays["global_features"][start : start + len(patches)] = tokens[:, 0]
                else:
                    arrays["target_tokens_aug"][start : start + len(patches), view - 1] = tokens
        target_seconds = time.perf_counter() - target_started
        neighbor_started = time.perf_counter()
        pending_images, pending_indices = [], []

        def flush():
            if not pending_images:
                return
            tokens = encode(torch.stack(pending_images))
            for row, (spot_index, grid_index) in enumerate(pending_indices):
                arrays["neighbor_features"][spot_index, grid_index] = tokens[row, 0]
            pending_images.clear()
            pending_indices.clear()

        for spot_index, center in enumerate(coords):
            for grid_index, (x, y, _, _) in enumerate(image_grid_boxes(center)):
                if grid_index == 12:
                    arrays["neighbor_features"][spot_index, grid_index] = arrays["global_features"][spot_index]
                elif masks[spot_index, grid_index]:
                    patch = slide.read_region((int(x), int(y)), 0, (224, 224)).convert("RGB")
                    pending_images.append(transform(patch))
                    pending_indices.append((spot_index, grid_index))
                    if len(pending_images) == batch_size:
                        flush()
                else:
                    # Native preprocessing leaves invalid tiles as zeros in
                    # NORMALIZED tensor space, then encodes them. A zero feature
                    # vector or normalized black PIL image would be different.
                    arrays["neighbor_features"][spot_index, grid_index] = zero_normalized_cls
            if (spot_index + 1) % 128 == 0:
                print(
                    f"NEIGHBORS {slide_id} {spot_index + 1}/{n} elapsed={time.perf_counter() - neighbor_started:.1f}",
                    flush=True,
                )
        flush()
        neighbor_seconds = time.perf_counter() - neighbor_started
        for array in arrays.values():
            array.flush()
        slide.close()
        files = {name: f"{name}.npy" for name in (*arrays, "positions", "spot_ids", "neighbor_mask")}
        checksums = {name: digest(output / relative) for name, relative in files.items()}
        metadata = {
            "schema": SCHEMA,
            "upstream_commit": plan["upstream_commit"],
            "slide_id": slide_id,
            "encoder": "Virchow2",
            "encoder_revision": ENCODER_REVISION,
            "model_sha256": model_hash,
            "preprocessing": {
                "crop": plan["crop"],
                "transforms": str(transform),
                "precision": plan["feature_storage"],
                "masked_neighbor_input": "zero normalized image tensor",
                "target_boundary": (
                    "OpenSlide black RGB padding outside WSI, including fully outside targets; center grid token retained; coordinates unchanged"
                    if plan["allow_target_padding"] else "reject incomplete target"
                ),
                "padded_target_spots": record.get("padded_target_spots", 0),
                "fully_outside_target_spots": record.get("fully_outside_target_spots", 0),
            },
            "neighbor_layout": "image_grid_5x5",
            "target_representation": "full_sequence",
            "context_representation": "cls",
            "position_space": "st_lattice",
            "target_augmentation_names": plan["target_augmentation_names"],
            "arrays": files,
            "sha256": checksums,
            "spot_order_sha256": hashlib.sha256("\n".join(spot_ids).encode()).hexdigest(),
            "spots": n,
            "seconds": time.perf_counter() - start_time,
            "target_seconds": target_seconds,
            "neighbor_seconds": neighbor_seconds,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(marker, metadata)
        GeneDMLSlideCache(marker)
        print(f"CACHE COMPLETE {slide_id} spots={n} seconds={metadata['seconds']:.2f}", flush=True)


def main():
    import anndata
    import openslide
    from src.baselines.gene_dml_cross.model import UPSTREAM_COMMIT
    from src.baselines.gene_dml_cross.data import image_grid_boxes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True, help="One cohort directory")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--slides", nargs="+", required=True)
    parser.add_argument("--target-boundary", choices=("reject", "pad"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    records = []
    for slide_id in args.slides:
        a = anndata.read_h5ad(args.data_path / "st" / f"{slide_id}.h5ad", backed="r")
        coords = np.asarray(a.obsm["spatial"])
        a.file.close()
        wsi = openslide.OpenSlide(str(args.data_path / "wsis" / f"{slide_id}.tif"))
        width, height = wsi.dimensions
        wsi.close()
        boxes = np.asarray([image_grid_boxes(c)[12] for c in coords])
        partial = (boxes[:, 0] < 0) | (boxes[:, 1] < 0) | (boxes[:, 2] > width) | (boxes[:, 3] > height)
        outside = (boxes[:, 2] <= 0) | (boxes[:, 3] <= 0) | (boxes[:, 0] >= width) | (boxes[:, 1] >= height)
        records.append(dict(slide=slide_id, spots=len(coords), padded_target_spots=int(partial.sum()),
                            fully_outside_target_spots=int(outside.sum())))
    plan = dict(encoder_directory=str(args.model_dir), data_root=str(args.data_path),
        cache_root=str(args.cache_root), slides=records, extraction_shards=1,
        extraction_batch_size=args.batch_size, allow_target_padding=args.target_boundary == "pad",
        upstream_commit=UPSTREAM_COMMIT, crop="224 pixels at level 0; 5x5 image grid",
        feature_storage="float16", target_augmentation_names=["identity", "flip_left_right", "flip_top_bottom"])
    from types import SimpleNamespace
    extract(SimpleNamespace(shard=0, slide=None), plan)


if __name__ == "__main__":
    main()
