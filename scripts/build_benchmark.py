"""Materialize portable training configurations from fixed splits and verified panels."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("bleep", "stem", "strank", "histogene", "stflow", "merge", "m2tglgo", "gene_dml", "simplep2g", "simplep2g_cross")
REFERENCE_METHODS = ("bleep", "histogene", "merge", "m2tglgo")


def model_defaults(method, cohort, reference):
    suffix = "_reference" if reference else ""
    path = ROOT / "configs" / "models" / f"{method}{suffix}.yaml"
    config = yaml.safe_load(path.read_text())
    overrides = config.pop("cohort_overrides", {}).get(cohort, {})
    for key, value in overrides.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    splits = json.loads(args.splits.read_text())["splits"]
    protocol = yaml.safe_load((ROOT / "configs/protocols/benchmark.yaml").read_text())
    reference = getattr(args, "inputs", "virchow2") == "reference"
    methods = REFERENCE_METHODS if reference else METHODS
    if args.method and args.method not in methods:
        raise ValueError(f"Reference inputs are available for {', '.join(REFERENCE_METHODS)}")
    tasks = []
    configs = []
    # Validate every input before creating output files.
    for split in splits:
        d, fold = split["cohort"], split["fold"]
        if args.cohort and d != args.cohort:
            continue
        if args.fold is not None and fold != args.fold:
            continue
        panel_root = args.panels.resolve() / d / f"fold{fold}"
        info = json.loads((panel_root / "manifest.json").read_text())
        if (set(info["selection_slides"]) != set(split["training_slides"])
                or info["validation_slide"] != split["validation_slide"]
                or info["test_slide"] != split["test_slide"]
                or info["validation_labels_used_in_selection"] or info["test_labels_used_in_selection"]):
            raise ValueError(f"Panel provenance does not match split: {panel_root}")
        for panel in ("hmhvg", "hvg"):
            genes = panel_root / f"selected_gene_list_{panel}.txt"
            slides = panel_root / "all_slide_lst.txt"
            gene_ids = genes.read_text().splitlines()
            if len(gene_ids) != 200 or len(set(gene_ids)) != 200:
                raise ValueError(f"Expected 200 distinct genes: {genes}")
            slide_ids = slides.read_text().splitlines()
            expected_slides = set(split["training_slides"]) | {split["validation_slide"]}
            if len(slide_ids) != len(expected_slides) or set(slide_ids) != expected_slides:
                raise ValueError(f"Unexpected loader slide list: {slides}")
            for path in (genes, slides):
                if digest(path) != info["sha256"][path.name]:
                    raise ValueError(f"Panel file changed: {path}")
            for method in methods:
                if args.method and method != args.method:
                    continue
                cfg = model_defaults(method, d, reference)
                variant = f"{method}_reference" if reference else method
                name = f"{d}_fold{fold}_{panel}_{variant}_seed42"
                cfg.update(config_stage="benchmark", cohort=d, fold=fold, panel=panel,
                    seed=42, total_epochs=int(protocol.get("epoch_overrides", {}).get(method, protocol["total_epochs"])),
                    training_slides=split["training_slides"],
                    validation_slide=split["validation_slide"], test_slide=split["test_slide"],
                    slide_out=split["validation_slide"],
                    data_path=str(args.data_root.resolve() / d) + "/",
                    gene_list_filename=str(genes), folder_list_filename=str(slides),
                    results_dir=str(output / "runs" / name), experiment_name=name,
                    checkpoint_selection="best_validation_pcc200", eval_filter_bad_spots=False)
                if method == "stem":
                    cfg.update(validation_num_eval_samples=1, num_eval_samples=20)
                cfg.update(protocol["precision_transitions"].get(name, {}))
                if method == "gene_dml":
                    cfg["model_config"]["checkpoint_global"] = d == "ccrcc"
                config = f"configs/{name}.yaml"
                configs.append((config, cfg))
                tasks.append(dict(split, name=name, method=method, panel=panel,
                                  config=config, output=f"runs/{name}", genes_sha256=digest(genes)))
    if not tasks:
        raise ValueError("No tasks match the requested cohort/fold/method")
    (output / "configs").mkdir(parents=True)
    for relative, cfg in configs:
        (output / relative).write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "manifest.json").write_text(json.dumps({"expected": len(tasks), "tasks": tasks}, indent=2) + "\n")
    print(f"Prepared {len(tasks)} configurations in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=ROOT / "configs/splits/heldout.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "hest1k_datasets")
    parser.add_argument("--panels", type=Path, default=ROOT / "results/panels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("brain", "prad", "her2st", "lung", "ccrcc"))
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--inputs", choices=("virchow2", "reference"), default="virchow2",
                        help="Use the shared Virchow2 inputs or the four paired reference inputs")
    build(parser.parse_args())
