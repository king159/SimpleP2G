"""Prepare pinned Virchow2 features and explicitly scoped gene panels for the benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def extract(args):
    import anndata
    import numpy as np
    import openslide
    import timm
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from timm.data import create_transform, resolve_data_config
    from timm.layers import SwiGLUPacked

    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction must run on an allocated GPU")
    torch.set_num_threads(min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.backends.cuda.matmul.allow_tf32 = False
    data = Path(args.data_root) / args.cohort
    proc = data / "processed_data"
    marker = proc / "feature_verified" / f"{args.slide}.json"
    outputs = [
        proc / "1spot_virchow2_ebd" / f"{args.slide}_virchow2.pt",
        proc / "1spot_virchow2_ebd_aug" / f"{args.slide}_virchow2_aug.pt",
    ]
    if marker.exists() and not args.limit:
        record = json.loads(marker.read_text())
        if record.get("model_revision") == REVISION and all(
            p.exists() and digest(p) == record["sha256"].get(str(p)) for p in outputs
        ):
            print("Verified existing features:", args.cohort, args.slide, flush=True)
            return
    adata = anndata.read_h5ad(data / "st" / f"{args.slide}.h5ad", backed="r")
    coords = np.asarray(adata.obsm["spatial"])
    spot_ids = list(map(str, adata.obs_names))
    spatial = adata.uns["spatial"]
    key = "ST" if "ST" in spatial else next(iter(spatial))
    diameter = float(spatial[key]["scalefactors"]["spot_diameter_fullres"])
    adata.file.close()
    radius = 112 if diameter < 224 else int(diameter // 2)
    if args.limit:
        coords = coords[: args.limit]
    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model = timm.create_model(
        config["architecture"],
        pretrained=False,
        pretrained_cfg=config["pretrained_cfg"],
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        **config["model_args"],
    )
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model = model.cuda().eval()
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    slide = openslide.OpenSlide(str(data / "wsis" / f"{args.slide}.tif"))
    base = torch.empty((len(coords), 2560), dtype=torch.float32)
    augmented = torch.empty((len(coords), 7, 2560), dtype=torch.float32)
    start_time = time.time()
    for start in range(0, len(coords), args.batch_size):
        patches = []
        for x, y in coords[start : start + args.batch_size]:
            # PIL.Image.crop rounds floating corners; preserve that convention.
            left, top, right, bottom = map(round, (x - radius, y - radius, x + radius, y + radius))
            patch = slide.read_region((left, top), 0, (right - left, bottom - top)).convert("RGB")
            patches.append(patch)
        for variant in range(8):
            images = patches if variant == 0 else [p.transpose(variant - 1) for p in patches]
            batch = torch.stack([transform(p.resize((224, 224), Image.Resampling.LANCZOS)) for p in images]).cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                tokens = model(batch)
                features = torch.cat((tokens[:, 0], tokens[:, 5:].mean(1)), dim=-1)
            if not torch.isfinite(features).all():
                raise RuntimeError(f"Non-finite features at spot {start}, variant {variant}")
            features = features.float().cpu()
            if variant == 0:
                base[start : start + len(patches)] = features
            else:
                augmented[start : start + len(patches), variant - 1] = features
        if start % (args.batch_size * 10) == 0:
            print(
                args.cohort,
                args.slide,
                start + len(patches),
                "/",
                len(coords),
                "seconds",
                round(time.time() - start_time, 1),
                flush=True,
            )
    slide.close()
    if args.limit:
        print("SMOKE OK", tuple(base.shape), tuple(augmented.shape), flush=True)
        return
    for path, tensor in zip(outputs, (base, augmented)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(tensor, temporary)
        temporary.replace(path)
    write_json(
        marker,
        {
            "cohort": args.cohort,
            "slide": args.slide,
            "spots": len(coords),
            "model_revision": REVISION,
            "model_sha256": digest(model_dir / "model.safetensors"),
            "precision": "fp16 inference, saved float32",
            "radius_level0_pixels": radius,
            "augmentation": "identity plus PIL transpose 0..6",
            "reader": "OpenSlide level 0",
            "sha256": {str(p): digest(p) for p in outputs},
            "spot_order_sha256": hashlib.sha256("\n".join(spot_ids).encode()).hexdigest(),
            "script_sha256": digest(__file__),
            "seconds": time.time() - start_time,
        },
    )
    print("FEATURES COMPLETE", args.cohort, args.slide, flush=True)


def select_panels(args):
    import anndata
    import numpy as np
    import pandas as pd
    import scanpy as sc

    splits = json.loads(Path(args.splits).read_text())["splits"]
    split = next(s for s in splits if s["cohort"] == args.cohort and s["fold"] == args.fold)
    included = sorted(split["training_slides"])
    test_slide = split["validation_slide"]
    if not included or len(set(included)) != len(included):
        raise ValueError("Training slide IDs must be distinct and nonempty")
    if test_slide == split["test_slide"] or set(included) & {test_slide, split["test_slide"]}:
        raise ValueError("Training, validation and test slides must be disjoint")
    slides = sorted(included + [test_slide])
    data = Path(args.data_root) / args.cohort
    dest = Path(args.output) / args.cohort / f"fold{args.fold}"
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing panels: {dest}")
    adatas = [anndata.read_h5ad(data / "st" / f"{s}.h5ad") for s in included]
    common = sorted(set.intersection(*(set(a.var_names) for a in adatas)))
    # Only feature identifiers, never expression values, may constrain the
    # shared measurement schema of an independent validation/test slide.
    schema_slides = [split["validation_slide"], split["test_slide"]]
    for schema_slide in schema_slides:
        schema = anndata.read_h5ad(data / "st" / f"{schema_slide}.h5ad", backed="r")
        common = sorted(set(common) & set(schema.var_names))
        schema.file.close()
    adatas = [a[:, common].copy() for a in adatas]
    candidates = set()
    for adata in adatas:
        a = adata.copy()
        sc.pp.filter_cells(a, min_genes=1)
        sc.pp.filter_genes(a, min_cells=1)
        sc.pp.normalize_total(a, inplace=True)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=2000)
        candidates.update(a.var_names[a.var["highly_variable"]])
    candidates = sorted(g for g in candidates if not g.startswith(("MT", "mt", "RPS", "RPL")))
    counts = np.concatenate([a[:, candidates].X.toarray() for a in adatas])
    frame = pd.DataFrame(counts, columns=candidates).fillna(0)
    ranks = frame.mean().rank(ascending=False, method="min") + frame.var().rank(ascending=False, method="min")
    hmhvg = sorted(ranks.sort_values().head(200).index)
    normalized = []
    for adata in adatas:
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        normalized.append(a)
    combined = anndata.concat(normalized, label="batch")
    sc.pp.highly_variable_genes(combined, n_top_genes=200, batch_key="batch")
    hvg = combined.var_names[combined.var["highly_variable"]].tolist()
    dest.mkdir(parents=True, exist_ok=True)
    test = anndata.read_h5ad(data / "st" / f"{test_slide}.h5ad", backed="r")
    for name, genes in (("hmhvg", hmhvg), ("hvg", hvg)):
        assert len(genes) == 200 and len(set(genes)) == 200
        assert set(genes).issubset(test.var_names), "Selected training genes absent from test feature schema"
        (dest / f"selected_gene_list_{name}.txt").write_text("\n".join(genes) + "\n")
    test.file.close()
    (dest / "all_slide_lst.txt").write_text("\n".join(slides) + "\n")

    write_json(dest / "manifest.json", {
        "selection_slides": included,
        "validation_slide": split["validation_slide"], "test_slide": split["test_slide"],
        "gene_schema_only_slides": schema_slides,
        "validation_labels_used_in_selection": False, "test_labels_used_in_selection": False,
        "script_sha256": digest(__file__),
        "sha256": {p.name: digest(p) for p in dest.glob("*.txt")},
    })
    print("PANELS COMPLETE", args.cohort, args.fold, len(included), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    features = sub.add_parser("features")
    features.add_argument("--model-dir", type=Path, required=True)
    features.add_argument("--slide", required=True)
    features.add_argument("--batch-size", type=int, default=64)
    features.add_argument("--limit", type=int, default=0)
    panels = sub.add_parser("panels")
    panels.add_argument("--splits", type=Path, default=Path("configs/splits/heldout.json"))
    panels.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    panels.add_argument("--output", type=Path, default=Path("results/panels"))
    for command in (features, panels):
        command.add_argument("--data-root", type=Path, default=Path("hest1k_datasets"))
        command.add_argument("--cohort", choices=("brain", "prad", "her2st", "lung", "ccrcc"), required=True)
    args = parser.parse_args()
    (extract if args.action == "features" else select_panels)(args)


if __name__ == "__main__":
    main()
