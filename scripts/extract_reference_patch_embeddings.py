#!/usr/bin/env python3
"""Extract reference encoder embeddings for paired-encoder controls.

This utility extracts images only and never recomputes gene panels. Cross-patch baselines consume one frozen embedding per
measured spot; single-patch controls may additionally request the seven
deterministic PIL transpose views used by the original training pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet18, resnet50
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hest_wsi_dataset import OpenSlideReader


Image.MAX_IMAGE_PIXELS = None


class ImageNetResNet50(nn.Module):
    output_dim = 2048
    name = "resnet50"

    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.transform = weights.transforms()

    def forward(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(device)
        return self.model(batch)


class MERGEResNet18(nn.Module):
    """Released TCGA-pretrained 256-d ResNetSimCLR encoder from MERGE."""

    output_dim = 256
    name = "merge_resnet18"

    def __init__(self, checkpoint: Path):
        super().__init__()
        backbone = resnet18(weights=None, norm_layer=nn.InstanceNorm2d)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.l1 = nn.Linear(backbone.fc.in_features, backbone.fc.in_features)
        self.l2 = nn.Linear(backbone.fc.in_features, self.output_dim)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MERGE checkpoint does not exist: {checkpoint}; pass --merge-checkpoint")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.load_state_dict(state, strict=True)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, images: list[Image.Image], device: torch.device) -> torch.Tensor:
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(device)
        features = self.features(batch).flatten(1)
        return self.l2(torch.relu(self.l1(features)))


def _spot_radius(adata) -> int:
    spatial = adata.uns.get("spatial", {})
    if not spatial:
        return 112
    library = next(iter(spatial.values()))
    diameter = float(library.get("scalefactors", {}).get("spot_diameter_fullres", 224))
    return 112 if diameter < 224 else int(diameter // 2)


def _wsi_path(data_path: Path, slide_id: str) -> Path:
    for suffix in (".tif", ".tiff", ".svs", ".png", ".jpg", ".jpeg"):
        candidate = data_path / "wsis" / f"{slide_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No WSI found for {slide_id} under {data_path / 'wsis'}")


def _extract_slide(
    encoder: nn.Module,
    reader: OpenSlideReader,
    coords,
    radius: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for start in tqdm(range(0, len(coords), batch_size), leave=False, desc=encoder.name):
        batch_coords = coords[start : start + batch_size]
        patches = [
            reader.read_region_rgb(
                int(float(x) - radius),
                int(float(y) - radius),
                2 * radius,
            )
            for x, y in batch_coords
        ]
        with torch.inference_mode():
            output = encoder(patches, device)
        outputs.append(output.detach().float().cpu())
    if not outputs:
        return torch.empty(0, encoder.output_dim)
    return torch.cat(outputs, dim=0)


def _extract_slide_augmented(
    encoder: nn.Module,
    reader: OpenSlideReader,
    coords,
    radius: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs = []
    for start in tqdm(range(0, len(coords), batch_size), leave=False, desc=f"{encoder.name}-aug"):
        batch_coords = coords[start : start + batch_size]
        patches = [
            reader.read_region_rgb(
                int(float(x) - radius),
                int(float(y) - radius),
                2 * radius,
            )
            for x, y in batch_coords
        ]
        augmented = [patch.transpose(transpose) for patch in patches for transpose in range(7)]
        with torch.inference_mode():
            output = encoder(augmented, device)
        outputs.append(output.detach().float().cpu().reshape(len(patches), 7, encoder.output_dim))
    if not outputs:
        return torch.empty(0, 7, encoder.output_dim)
    return torch.cat(outputs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["resnet50", "merge_resnet18"],
        default=["resnet50", "merge_resnet18"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="PyTorch CPU threads per extractor; one avoids transform oversubscription.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-augmentations",
        action="store_true",
        help="Also save the seven deterministic PIL transpose views used by single-patch training.",
    )
    parser.add_argument(
        "--merge-checkpoint",
        type=Path,
        default=Path("checkpoints/merge/model-low-v1.pth"),
        help="Released MERGE ResNet18 weights, stored independently of the source checkout",
    )
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if "merge_resnet18" in args.encoders and not args.merge_checkpoint.is_file():
        parser.error("MERGE requires its released weights: --merge-checkpoint /path/to/model-low-v1.pth")
    torch.set_num_threads(max(1, args.cpu_threads))
    device = torch.device(args.device)
    encoders: list[nn.Module] = []
    if "resnet50" in args.encoders:
        encoders.append(ImageNetResNet50())
    if "merge_resnet18" in args.encoders:
        encoders.append(MERGEResNet18(args.merge_checkpoint))
    for encoder in encoders:
        encoder.to(device).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)

    for data_path in args.data_path:
        st_dir = data_path / "st"
        slide_paths = sorted(st_dir.glob("*.h5ad"))[args.shard_index :: args.num_shards]
        for h5ad_path in slide_paths:
            slide_id = h5ad_path.stem
            pending = []
            for encoder in encoders:
                base_path = (
                    data_path / "processed_data" / f"1spot_{encoder.name}_ebd" / f"{slide_id}_{encoder.name}.pt"
                )
                aug_path = (
                    data_path
                    / "processed_data"
                    / f"1spot_{encoder.name}_ebd_aug"
                    / f"{slide_id}_{encoder.name}_aug.pt"
                )
                need_base = args.overwrite or not base_path.exists()
                need_aug = args.include_augmentations and (args.overwrite or not aug_path.exists())
                if need_base or need_aug:
                    pending.append((encoder, base_path, aug_path, need_base, need_aug))
            if not pending:
                print(f"[skip] {data_path.name}/{slide_id}: all requested embeddings exist")
                continue

            adata = anndata.read_h5ad(h5ad_path, backed="r")
            coords = adata.obsm["spatial"]
            radius = _spot_radius(adata)
            reader = OpenSlideReader(str(_wsi_path(data_path, slide_id)))
            try:
                for encoder, base_path, aug_path, need_base, need_aug in pending:
                    if need_base:
                        print(
                            f"[extract] {data_path.name}/{slide_id} encoder={encoder.name} "
                            f"spots={len(coords)} radius={radius}"
                        )
                        embeddings = _extract_slide(
                            encoder,
                            reader,
                            coords,
                            radius,
                            args.batch_size,
                            device,
                        )
                        if embeddings.shape != (adata.n_obs, encoder.output_dim):
                            raise RuntimeError(
                                f"Unexpected shape for {slide_id}/{encoder.name}: {tuple(embeddings.shape)}"
                            )
                        base_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(embeddings, base_path)
                    if need_aug:
                        print(
                            f"[extract] {data_path.name}/{slide_id} encoder={encoder.name}-aug "
                            f"spots={len(coords)} radius={radius}"
                        )
                        embeddings_aug = _extract_slide_augmented(
                            encoder,
                            reader,
                            coords,
                            radius,
                            args.batch_size,
                            device,
                        )
                        if embeddings_aug.shape != (adata.n_obs, 7, encoder.output_dim):
                            raise RuntimeError(
                                f"Unexpected augmented shape for {slide_id}/{encoder.name}: "
                                f"{tuple(embeddings_aug.shape)}"
                            )
                        aug_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(embeddings_aug, aug_path)
            finally:
                if reader._slide is not None:
                    reader._slide.close()


if __name__ == "__main__":
    main()
