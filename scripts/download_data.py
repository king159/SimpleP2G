"""Pin, download, and verify the five benchmark cohorts.

Uses the existing Hugging Face login. Credentials are never written to manifests.
Run `plan` once, then `download --dataset brain` (resumable).
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "MahmoodLab/hest"
EXPECTED = {"brain": 14, "prad": 23, "her2st": 36, "lung": 20, "ccrcc": 24}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def select_cohorts(metadata):
    with Path(metadata).open() as handle:
        rows = list(csv.DictReader(handle))
    selectors = {
        "brain": lambda r: r["dataset_title"].startswith("Spatial Multimodal Analysis")
        and r["disease_state"] == "Healthy",
        "prad": lambda r: r["id"] in {f"MEND{i}" for i in range(139, 163)},
        "her2st": lambda r: r["dataset_title"].startswith("Spatial deconvolution of"),
        "lung": lambda r: r["dataset_title"].startswith("Image-based spatial transcriptomics"),
        "ccrcc": lambda r: r["dataset_title"].startswith(
            "Tertiary lymphoid structures generate and propagate anti-tumor antibody-producing plasma cells in renal cell cancer"
        ),
    }
    cohorts = {}
    for name, select in selectors.items():
        selected = sorted((r for r in rows if select(r)), key=lambda r: r["id"])
        ids = [r["id"] for r in selected]
        if len(ids) != EXPECTED[name] or len(set(ids)) != len(ids):
            raise ValueError(f"Unexpected cohort membership for {name}: {ids}")
        cohorts[name] = selected
    return cohorts


def plan(args):
    from huggingface_hub import HfApi, hf_hub_download

    if args.manifest.exists():
        raise FileExistsError(f"Manifest already exists; reuse its pinned revision: {args.manifest}")
    api = HfApi(endpoint="https://huggingface.co")
    revision = api.dataset_info(REPO_ID, revision=args.revision).sha
    metadata = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                    filename="HEST_v1_2_1.csv", revision=revision, endpoint=api.endpoint))
    cohorts = select_cohorts(metadata)
    samples = []
    for dataset, rows in cohorts.items():
        for row in rows:
            samples.append({
                "dataset": dataset, "slide_id": row["id"], "patient": row["patient"] or None,
                "study": row["dataset_title"], "species": row["species"], "platform": row["st_technology"],
            })
    paths = [f"{folder}/{sample['slide_id']}{suffix}" for sample in samples
             for folder, suffix in [("st", ".h5ad"), ("wsis", ".tif"), ("metadata", ".json")]]
    info = {}
    for start in range(0, len(paths), 50):
        for entry in api.get_paths_info(REPO_ID, paths[start:start + 50], repo_type="dataset", revision=revision):
            lfs = getattr(entry, "lfs", None)
            info[entry.path] = {"path": entry.path, "bytes": entry.size,
                                "sha256": getattr(lfs, "sha256", None), "git_blob": entry.blob_id}
    missing = sorted(set(paths) - set(info))
    if missing:
        raise FileNotFoundError(f"Required files missing at {revision}: {missing}")
    for sample in samples:
        sample["files"] = [info[f"{folder}/{sample['slide_id']}{suffix}"]
                           for folder, suffix in [("st", ".h5ad"), ("wsis", ".tif"), ("metadata", ".json")]]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "repo_id": REPO_ID, "revision": revision,
        "metadata_file": metadata.name, "metadata_sha256": sha256(metadata),
        "data_root": str((args.data_root or ROOT / "hest1k_datasets").resolve()), "samples": samples,
        "notes": ["Metadata fixes cohort membership to the submitted five datasets.",
                  "Raw files are pinned to the recorded HF revision; this does not establish historical byte identity.",
                  "Patient is unavailable for Brain in this metadata; do not claim patient-independent Brain folds."],
    }
    write_json(args.manifest, manifest)
    for dataset in EXPECTED:
        group = [s for s in samples if s["dataset"] == dataset]
        size = sum(f["bytes"] for s in group for f in s["files"])
        print(f"{dataset}: {len(group)} slides, {size / 1024**3:.2f} GiB", flush=True)
    total = sum(f["bytes"] for s in samples for f in s["files"])
    print(f"Total: {total / 1024**3:.2f} GiB; free: {shutil.disk_usage(ROOT).free / 1024**3:.2f} GiB")
    print(f"Pinned manifest: {args.manifest}")


def download(args):
    from huggingface_hub import hf_hub_download

    manifest = json.loads(args.manifest.read_text())
    data_root = args.data_root.resolve() if args.data_root is not None else Path(manifest["data_root"])
    samples = [s for s in manifest["samples"] if args.dataset == "all" or s["dataset"] == args.dataset]
    if args.slide:
        samples = [s for s in samples if s["slide_id"] in args.slide]
        if {s["slide_id"] for s in samples} != set(args.slide):
            raise ValueError("Some requested slides are outside the selected cohort")

    def already_verified(sample):
        local_dir = data_root / sample["dataset"]
        marker = local_dir / "download_verified" / (sample["slide_id"] + ".json")
        try:
            checked = json.loads(marker.read_text())
            if checked.get("revision") != manifest["revision"]:
                return False
            records = {item["path"]: item for item in checked["files"]}
            for item in sample["files"]:
                path = local_dir / item["path"]
                record = records[item["path"]]
                if path.stat().st_size != item["bytes"] or record["bytes"] != item["bytes"]:
                    return False
                if path.stat().st_mtime_ns > marker.stat().st_mtime_ns:
                    return False
                if item["sha256"] and record["sha256"] != item["sha256"]:
                    return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    selected_count = len(samples)
    samples = [sample for sample in samples if not already_verified(sample)]
    skipped_count = selected_count - len(samples)
    print(f"Previously verified: {skipped_count}; remaining: {len(samples)}", flush=True)
    needed = sum(f["bytes"] for s in samples for f in s["files"]
                 if not (data_root / s["dataset"] / f["path"]).is_file())
    data_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(data_root).free < needed + 20 * 1024**3:
        raise OSError("Insufficient free space for the selected files plus a 20 GiB reserve")

    def one_sample(sample):
        local_dir = data_root / sample["dataset"]
        completed = []
        for item in sample["files"]:
            force_download = False
            for attempt in range(1, args.retries + 1):
                try:
                    print(f"Downloading {sample['dataset']}/{item['path']} (attempt {attempt})", flush=True)
                    path = Path(hf_hub_download(manifest["repo_id"], item["path"], repo_type="dataset",
                                revision=manifest["revision"], local_dir=local_dir,
                                endpoint="https://huggingface.co", force_download=force_download))
                    if path.stat().st_size != item["bytes"]:
                        force_download = True
                        raise ValueError(f"Size mismatch: {path}")
                    digest = sha256(path)
                    if item["sha256"] and digest != item["sha256"]:
                        force_download = True
                        raise ValueError(f"SHA-256 mismatch: {path}")
                    break
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    reason = type(exc).__name__ + (f" HTTP {status}" if status else "")
                    print(f"Download error: {item['path']}: {reason}", flush=True)
                    if attempt == args.retries:
                        raise RuntimeError(f"{item['path']}: {reason} after {attempt} attempts") from None
                    time.sleep(min(2 ** attempt, 30))
            completed.append({"path": item["path"], "bytes": item["bytes"], "sha256": digest})
        write_json(local_dir / "download_verified" / (sample["slide_id"] + ".json"),
                   {"revision": manifest["revision"], "files": completed})
        return sample["dataset"], sample["slide_id"]

    status = {"state": "running", "selected_slides": selected_count,
              "verified_slides": skipped_count, "failures": [], "pid": os.getpid()}
    status_path = args.manifest.parent / "download_status.json"
    write_json(status_path, status)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_sample, sample): sample for sample in samples}
        for future in as_completed(futures):
            sample = futures[future]
            try:
                dataset, slide = future.result()
                status["verified_slides"] += 1
                print(f"[{status['verified_slides']}/{selected_count}] Verified {dataset}/{slide}", flush=True)
            except Exception as exc:
                status["failures"].append({"dataset": sample["dataset"], "slide": sample["slide_id"],
                                           "error": str(exc)})
                print(f"Failed {sample['dataset']}/{sample['slide_id']}: {exc}", flush=True)
            write_json(status_path, status)
    status["state"] = "failed" if status["failures"] else "complete"
    write_json(status_path, status)
    if status["failures"]:
        raise SystemExit(f"{len(status['failures'])} slides failed; rerun to resume. See {status_path}")
    print(f"Complete: {status['verified_slides']}/{selected_count} slides verified", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "download"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/data_manifest.json")
    parser.add_argument("--data-root", type=Path, help="Override the manifest destination when relocating a reproduction")
    parser.add_argument("--revision", default="7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec")
    parser.add_argument("--dataset", choices=["all", *EXPECTED], default="all")
    parser.add_argument("--slide", nargs="+")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.retries < 1:
        parser.error("--retries must be positive")
    (plan if args.command == "plan" else download)(args)


if __name__ == "__main__":
    main()
