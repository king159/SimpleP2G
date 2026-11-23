from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WSIPatchConfig:
    patch_size: int = 224
    level: int = 0
    imagenet_norm: bool = True


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    return TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class OpenSlideReader:
    def __init__(self, wsi_path: str):
        self.wsi_path = wsi_path
        self._slide = None

    def _get_slide(self):
        if self._slide is None:
            import openslide

            self._slide = openslide.OpenSlide(self.wsi_path)
        return self._slide

    def __getstate__(self):
        # Allow pickling by resetting _slide to None (OpenSlide object is not picklable)
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def read_region_rgb(self, x: int, y: int, size: int, level: int = 0) -> Image.Image:
        slide = self._get_slide()
        img = slide.read_region((int(x), int(y)), int(level), (int(size), int(size)))
        return img.convert("RGB")


class HESTWSISpotDataset(Dataset):
    """On-the-fly patch extraction from HEST1k WSIs + gene vector from .h5ad.

    Returns a tuple `(x, y)`:
      - `x`: (G,) float32 gene vector (already log2(x+1))
      - `y`: dict with keys:
          - `image`: (3, H, W) float32 tensor
          - `coords`: (2,) int64 tensor (pixel coords in fullres: [x, y])
          - `slide`: str slide id (kept as python string; collate will create list)
    """

    def __init__(
        self,
        *,
        slide_ids: Iterable[str],
        data_path: str,
        selected_genes: list[str],
        patch_cfg: WSIPatchConfig,
        train: bool,
        n_pos_bins: int = 64,
    ):
        super().__init__()
        self.slide_ids = list(slide_ids)
        self.data_path = data_path
        self.selected_genes = selected_genes
        self.patch_cfg = patch_cfg
        self.train = train
        self.n_pos_bins = int(n_pos_bins)

        self._slides: dict[str, OpenSlideReader] = {}

        # Build flat index mapping: global_idx -> (slide_id, spot_idx)
        self._index: list[tuple[str, int]] = []
        self._expr: dict[str, np.ndarray] = {}
        self._coords: dict[str, np.ndarray] = {}
        self._pos_bins: dict[str, np.ndarray] = {}

        import anndata  # local import; env-specific

        for sid in self.slide_ids:
            adata_path = os.path.join(self.data_path, "st", f"{sid}.h5ad")
            adata = anndata.read_h5ad(adata_path)

            # genes: dense array for selected genes
            X_any: Any = adata[:, self.selected_genes].X
            toarray = getattr(X_any, "toarray", None)
            X_np = toarray() if callable(toarray) else np.asarray(X_any)
            X_np = np.asarray(X_np, dtype=np.float32)
            X_np = np.log2(X_np + 1.0)

            # coords: use fullres pixel coords; (N,2) as [x,y]
            if "spatial" in adata.obsm:
                coords = np.asarray(adata.obsm["spatial"]).astype(np.int64)
            else:
                coords = np.stack(
                    [
                        np.asarray(adata.obs["pxl_col_in_fullres"], dtype=np.int64),
                        np.asarray(adata.obs["pxl_row_in_fullres"], dtype=np.int64),
                    ],
                    axis=1,
                )

            # discretize coords for coordinate embeddings (HisToGene-style)
            pos_bins = self._discretize_coords(coords, n_bins=self.n_pos_bins)

            self._expr[sid] = X_np
            self._coords[sid] = coords
            self._pos_bins[sid] = pos_bins

            self._index.extend([(sid, i) for i in range(coords.shape[0])])

    @staticmethod
    def _discretize_coords(coords_xy: np.ndarray, *, n_bins: int) -> np.ndarray:
        coords_xy = coords_xy.astype(np.float32)
        mins = coords_xy.min(axis=0, keepdims=True)
        maxs = coords_xy.max(axis=0, keepdims=True)
        denom = np.maximum(maxs - mins, 1.0)
        norm = (coords_xy - mins) / denom
        bins = np.rint(norm * (n_bins - 1)).astype(np.int64)
        bins = np.clip(bins, 0, n_bins - 1)
        return bins

    def __len__(self) -> int:
        return len(self._index)

    def _get_reader(self, slide_id: str) -> OpenSlideReader:
        reader = self._slides.get(slide_id)
        if reader is None:
            wsi_path = os.path.join(self.data_path, "wsis", f"{slide_id}.tif")
            reader = OpenSlideReader(wsi_path)
            self._slides[slide_id] = reader
        return reader

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # PIL -> float32 tensor in [0,1]
        if self.train:
            if torch.rand(()) > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if torch.rand(()) > 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            angle = int(np.random.choice([0, 90, 180, 270]))
            if angle:
                img = img.rotate(angle)

        x = TF.to_tensor(img)
        if self.patch_cfg.imagenet_norm:
            x = _imagenet_normalize(x)
        return x

    def __getitem__(self, idx: int):
        slide_id, spot_idx = self._index[idx]

        x = torch.from_numpy(self._expr[slide_id][spot_idx]).float()
        coords_xy = self._coords[slide_id][spot_idx]
        pos_bins = self._pos_bins[slide_id][spot_idx]

        ps = int(self.patch_cfg.patch_size)
        half = ps // 2
        cx, cy = int(coords_xy[0]), int(coords_xy[1])

        reader = self._get_reader(slide_id)
        img = reader.read_region_rgb(cx - half, cy - half, ps, level=self.patch_cfg.level)
        img_t = self._transform(img)

        y = {
            "image": img_t,
            "coords": torch.tensor([cx, cy], dtype=torch.long),
            "pos_bins": torch.from_numpy(pos_bins).long(),
            "slide": slide_id,
        }
        return x, y
