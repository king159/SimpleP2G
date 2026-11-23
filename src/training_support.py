"""Datasets, model construction and metrics for independent-validation training."""
import math

import anndata
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from tqdm import tqdm

from src.baselines.bleep.api import (
    build_bleep_any_from_omegaconf,
    is_bleep_model,
    is_bleep_original,
)
from src.baselines.histogene.model import build_histogene_from_omegaconf, build_histogene_original_from_omegaconf
from src.baselines.stem.model import StemModel
from src.data.hest_wsi_dataset import HESTWSISpotDataset, WSIPatchConfig
from src.data.histogene_dataset import HisToGeneSlideDataset
from src.simpleP2G import SimpleP2G
from src.simpleP2G_neigh import SimpleP2GNeigh


# Mapping of encoder names to (dimension, ebd_dir, aug_dir, suffix)
ENCODER_REGISTRY = {
    "virchow2": (2560, "1spot_virchow2_ebd", "1spot_virchow2_ebd_aug", "virchow2"),
    "uni": (1024, "1spot_uni_ebd", "1spot_uni_ebd_aug", "uni"),
    "conch": (512, "1spot_conch_ebd", "1spot_conch_ebd_aug", "conch"),
    "dinov2_small": (384, "1spot_dinov2_small_ebd", "1spot_dinov2_small_ebd_aug", "dinov2_small"),
    "dinov2_large": (1024, "1spot_dinov2_large_ebd", "1spot_dinov2_large_ebd_aug", "dinov2_large"),
    "sam_base": (256, "1spot_sam_base_ebd", "1spot_sam_base_ebd_aug", "sam_base"),
    "sam_large": (256, "1spot_sam_large_ebd", "1spot_sam_large_ebd_aug", "sam_large"),
    "resnet50": (2048, "1spot_resnet50_ebd", "1spot_resnet50_ebd_aug", "resnet50"),
    "merge_resnet18": (
        256,
        "1spot_merge_resnet18_ebd",
        "1spot_merge_resnet18_ebd_aug",
        "merge_resnet18",
    ),
    "dinov3_large": (1024, "1spot_dinov3_large_ebd", "1spot_dinov3_large_ebd_aug", "dinov3_large"),
}

CROSS_PATCH_SIMPLE_MODELS = {"simplep2g_cross"}



def _get_encoder_dim(encoder_name: str) -> int:
    """Return embedding dimension for a given encoder name."""
    if encoder_name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. Supported: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[encoder_name][0]


def _load_slide_embeddings(data_path: str, sample_name: str, encoder_name: str, load_aug: bool = True):
    """Load precomputed embeddings for a single slide.

    Returns:
        ebd: (N, D) tensor
        ebd_aug: (N, 7, D) tensor or None if load_aug=False
    """
    proc = data_path + "processed_data/"
    _dim, ebd_dir, aug_dir, suffix = ENCODER_REGISTRY[encoder_name]
    ebd = torch.load(proc + ebd_dir + "/" + sample_name + "_" + suffix + ".pt", map_location="cpu")
    ebd_aug = None
    if load_aug:
        ebd_aug = torch.load(proc + aug_dir + "/" + sample_name + "_" + suffix + "_aug.pt", map_location="cpu")
    return ebd, ebd_aug


def customized_lr_scheduler(optimizer, warmup_steps=-1, total_steps=-1, min_lr_ratio=0.1):
    """LR scheduler with linear warmup and cosine decay.

    When cosine decay is disabled, use a constant rate after optional warmup.
    """

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        if total_steps > 0:
            # cosine decay from 1.0 to min_lr_ratio
            progress = (step - max(warmup_steps, 0)) / max(total_steps - max(warmup_steps, 0), 1)
            progress = min(progress, 1.0)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, fn)



class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def evaluate_one_step(pred, test) -> dict[str, float]:
    # calculate correlation per gene
    all_corr = []
    for i in range(test.shape[1]):
        x = test[:, i]
        y = pred[:, i]
        cor = np.corrcoef(x, y)[0][1]
        all_corr.append(cor)

    # metrics
    all_corr_sorted = np.array(sorted([c for c in all_corr if not np.isnan(c)]))
    all_corr_sorted = all_corr_sorted[::-1]

    metrics = {
        "PCC10": float(all_corr_sorted[:10].mean()),
        "PCC50": float(all_corr_sorted[:50].mean()),
        "PCC200": float(all_corr_sorted[:200].mean()),
        "MSE": float(np.mean((test - pred) ** 2)),
        "MAE": float(np.mean(np.abs(test - pred))),
    }

    return metrics


def _assemble_slide_embedding_dataset(input_args, logger, slidename_lst, selected_genes):
    """Per-slide dict dataset shared by STFlow / MERGE baselines."""
    from src.data.slide_embedding_dataset import SlideEmbeddingDataset

    enc_name = str(input_args.image_encoder)
    if enc_name not in ENCODER_REGISTRY:
        raise ValueError(f"STFlow/MERGE require a registered encoder, got {enc_name!r}")
    _dim, ebd_dir, _aug_dir, suffix = ENCODER_REGISTRY[enc_name]
    common_kw = dict(
        data_path=str(getattr(input_args, "data_path")),
        selected_genes=selected_genes,
        encoder_name=enc_name,
        ebd_dir=ebd_dir,
        ebd_suffix=suffix,
    )
    train_dataset = SlideEmbeddingDataset(
        slide_ids=slidename_lst,
        train=True,
        max_spots=int(getattr(input_args, "max_spots", 0)) or None,
        **common_kw,
    )
    eval_dataset = SlideEmbeddingDataset(
        slide_ids=input_args.slide_out.split(","),
        train=False,
        **common_kw,
    )
    return train_dataset, eval_dataset


def _assemble_neigh_slide_dataset(input_args, logger, slidename_lst, selected_genes):
    """Slide-backed dataset for simpleP2G-cross using one image encoder."""
    from src.data.neigh_slide_dataset import NeighSlideDataset

    enc_name = str(input_args.image_encoder)
    if enc_name not in ENCODER_REGISTRY:
        raise ValueError(f"simpleP2G-cross requires a registered encoder, got {enc_name!r}")
    _dim, ebd_dir, aug_dir, suffix = ENCODER_REGISTRY[enc_name]
    if str(getattr(input_args, "context_embedding_source", "target_encoder")) != "target_encoder":
        raise ValueError("simpleP2G-cross requires context_embedding_source='target_encoder'")
    input_args.neighbor_cond_size = _dim
    input_args.global_cond_size = _dim
    common_kw = dict(
        data_path=str(getattr(input_args, "data_path")),
        selected_genes=selected_genes,
        ebd_dir=ebd_dir,
        ebd_suffix=suffix,
        aug_dir=aug_dir,
        aug_suffix=suffix,
        neighbor_dim=_dim,
        num_neighbor_tokens=int(getattr(input_args, "num_neighbor_tokens", 24)),
        filter_bad_spots=bool(getattr(input_args, "filter_bad_spots", False)),
        chunks_per_slide=int(getattr(input_args, "chunks_per_slide", 1)),
        num_aug_ratio=int(getattr(input_args, "num_aug_ratio", 0)),
        sample_augmented_targets=bool(getattr(input_args, "sample_augmented_targets", False)),
    )
    train_dataset = NeighSlideDataset(
        slide_ids=slidename_lst,
        train=True,
        max_spots=int(getattr(input_args, "max_spots", 0)) or None,
        **common_kw,
    )
    eval_dataset = NeighSlideDataset(
        slide_ids=input_args.slide_out.split(","),
        train=False,
        **{
            **common_kw,
            "filter_bad_spots": bool(input_args.eval_filter_bad_spots),
        },
    )
    logger.info(
        "NeighSlideDataset -- train items: %d eval slides: %d max_spots: %s "
        "filter_bad_spots: %s chunks_per_slide: %s sample_augmented_targets: %s "
        "context_source: %s target/local/slide: %s/%s/%s",
        len(train_dataset),
        len(eval_dataset),
        getattr(input_args, "max_spots", None),
        bool(getattr(input_args, "filter_bad_spots", False)),
        int(getattr(input_args, "chunks_per_slide", 1)),
        bool(getattr(input_args, "sample_augmented_targets", False)),
        "target_encoder",
        bool(getattr(input_args, "use_target_context", True)),
        bool(getattr(input_args, "use_neighbor_context", True)),
        bool(getattr(input_args, "use_slide_context", False)),
    )
    return train_dataset, eval_dataset


def assemble_dataset(input_args, logger):
    model_name = str(getattr(input_args, "model", "")).lower()

    # ── Cross-patch embedding baselines ── per-slide dict batches
    if model_name in {"stflow", "merge", "m2tglgo"}:
        slidename_lst = list(np.genfromtxt(input_args.folder_list_filename, dtype=str))
        for slide_out in input_args.slide_out.split(","):
            slidename_lst.remove(slide_out)
            logger.info(f"{slide_out} is held out for validation.")
        logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")
        selected_genes = list(np.genfromtxt(input_args.gene_list_filename, dtype=str))
        input_args.input_gene_size = len(selected_genes)
        logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")
        return _assemble_slide_embedding_dataset(input_args, logger, slidename_lst, selected_genes)

    # ── simpleP2G-neigh slide-level comparable variant ── per-slide dict batches
    if model_name in CROSS_PATCH_SIMPLE_MODELS:
        slidename_lst = list(np.genfromtxt(input_args.folder_list_filename, dtype=str))
        for slide_out in input_args.slide_out.split(","):
            slidename_lst.remove(slide_out)
            logger.info(f"{slide_out} is held out for validation.")
        logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")
        selected_genes = list(np.genfromtxt(input_args.gene_list_filename, dtype=str))
        input_args.input_gene_size = len(selected_genes)
        logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")
        return _assemble_neigh_slide_dataset(input_args, logger, slidename_lst, selected_genes)

    # ── HisToGene (both modes) ── slide-level batching with cross-spot attention
    if model_name in {"histogene", "histogene_enhanced", "histogene_original"}:
        slidename_lst = list(np.genfromtxt(input_args.folder_list_filename, dtype=str))
        for slide_out in input_args.slide_out.split(","):
            slidename_lst.remove(slide_out)
            logger.info(f"{slide_out} is held out for validation.")
        logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

        selected_genes = list(np.genfromtxt(input_args.gene_list_filename, dtype=str))
        input_args.input_gene_size = len(selected_genes)
        logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")

        histo_cfg = getattr(input_args, "histogene", None)
        n_pos_bins = int(getattr(histo_cfg, "n_pos", 64)) if histo_cfg else 64
        max_spots = int(getattr(input_args, "max_spots", 0)) or None  # 0 means unlimited

        use_emb = model_name in {"histogene", "histogene_enhanced"}
        data_path = str(getattr(input_args, "data_path"))

        common_kw = {
            "data_path": data_path,
            "selected_genes": selected_genes,
            "use_embeddings": use_emb,
            "n_pos_bins": n_pos_bins,
        }
        if use_emb:
            common_kw["encoder_name"] = str(input_args.image_encoder)
        else:
            common_kw["patch_cfg"] = WSIPatchConfig(
                patch_size=int(getattr(input_args, "patch_size", 112)),
                level=int(getattr(input_args, "wsi_level", 0)),
                imagenet_norm=bool(getattr(input_args, "imagenet_norm", True)),
            )

        train_dataset = HisToGeneSlideDataset(
            slide_ids=slidename_lst,
            train=True,
            max_spots=max_spots,
            **common_kw,
        )
        eval_dataset = HisToGeneSlideDataset(
            slide_ids=input_args.slide_out.split(","),
            train=False,
            **common_kw,
        )
        return train_dataset, eval_dataset

    # ── BLEEP original ── raw patch pixels + coordinates from WSI.
    if is_bleep_original(model_name):
        slidename_lst = list(np.genfromtxt(input_args.folder_list_filename, dtype=str))
        for slide_out in input_args.slide_out.split(","):
            slidename_lst.remove(slide_out)
            logger.info(f"{slide_out} is held out for validation.")
        logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

        selected_genes = list(np.genfromtxt(input_args.gene_list_filename, dtype=str))
        input_args.input_gene_size = len(selected_genes)
        logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")

        patch_cfg = WSIPatchConfig(
            patch_size=int(getattr(input_args, "patch_size", 224)),
            level=int(getattr(input_args, "wsi_level", 0)),
            imagenet_norm=bool(getattr(input_args, "imagenet_norm", True)),
        )

        n_pos_bins = 64

        train_dataset = HESTWSISpotDataset(
            slide_ids=slidename_lst,
            data_path=str(getattr(input_args, "data_path")),
            selected_genes=selected_genes,
            patch_cfg=patch_cfg,
            train=True,
            n_pos_bins=n_pos_bins,
        )

        eval_dataset = HESTWSISpotDataset(
            slide_ids=input_args.slide_out.split(","),
            data_path=str(getattr(input_args, "data_path")),
            selected_genes=selected_genes,
            patch_cfg=patch_cfg,
            train=False,
            n_pos_bins=n_pos_bins,
        )
        return train_dataset, eval_dataset

    # load & assemble data
    slidename_lst = list(np.genfromtxt(input_args.folder_list_filename, dtype=str))
    # Leave the validation slide out of training.
    for slide_out in input_args.slide_out.split(","):
        slidename_lst.remove(slide_out)
        logger.info(f"{slide_out} is held out for validation.")
    logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

    # load selected gene list
    selected_genes = list(np.genfromtxt(input_args.gene_list_filename, dtype=str))
    input_args.input_gene_size = len(selected_genes)
    logger.info(f"Selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")


    if model_name == "strank":
        from src.baselines.strank.data import assemble_strank_datasets

        return assemble_strank_datasets(input_args, logger, slidename_lst, selected_genes, _load_slide_embeddings)

    enc_dim = _get_encoder_dim(input_args.image_encoder)
    all_img_ebd_aug = torch.empty(0, 7, enc_dim)
    all_img_ebd = torch.empty(0, enc_dim)
    all_count_mtx_ori = np.empty((0, len(selected_genes)))
    for sample_name in tqdm(slidename_lst, desc="Loading training slides"):
        train_adata = anndata.read_h5ad(input_args.data_path + "st/" + sample_name + ".h5ad")
        train_count_mtx = pd.DataFrame(
            train_adata[:, selected_genes].X.toarray(),
            columns=selected_genes,
            index=[sample_name + "_" + str(i) for i in range(train_adata.shape[0])],
        )
        slide_img_ebd, slide_img_ebd_aug = _load_slide_embeddings(
            input_args.data_path, sample_name, input_args.image_encoder, load_aug=True
        )
        all_img_ebd_aug = torch.cat([all_img_ebd_aug, slide_img_ebd_aug], axis=0)
        all_img_ebd = torch.cat([all_img_ebd, slide_img_ebd], axis=0)
        all_count_mtx_ori = np.concatenate((all_count_mtx_ori, train_count_mtx), axis=0)
    # randomly select augmented patches according to the input augmentation ratio (int)
    num_aug_ratio = input_args.num_aug_ratio
    all_count_mtx_aug = np.repeat(
        np.copy(all_count_mtx_ori), num_aug_ratio, axis=0
    )  # generate count matrix for all augmented patches
    selected_img_ebd_aug = torch.zeros((all_count_mtx_aug.shape[0], all_img_ebd_aug.shape[2]))
    for i in range(all_img_ebd_aug.shape[0]):  # randomly select augmented patches
        selected_transpose_idx = np.random.choice(all_img_ebd_aug.shape[1], num_aug_ratio, replace=False)
        selected_img_ebd_aug[i * num_aug_ratio : (i + 1) * num_aug_ratio, :] = all_img_ebd_aug[
            i, selected_transpose_idx, :
        ]

    all_img_ebd = torch.cat([all_img_ebd, selected_img_ebd_aug], axis=0)
    all_count_mtx = np.concatenate((all_count_mtx_ori, all_count_mtx_aug), axis=0)

    ################################################
    all_count_mtx_df = pd.DataFrame(all_count_mtx, columns=selected_genes, index=list(range(all_count_mtx.shape[0])))
    # remove the spot with all NAN/zero in count mtx
    all_count_mtx_all_nan_spot_index = all_count_mtx_df.index[all_count_mtx_df.isnull().all(axis=1)]
    all_count_mtx_all_zero_spot_index = all_count_mtx_df.index[all_count_mtx_df.sum(axis=1) == 0]
    spot_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
    spot_idx_to_keep = list(set(all_count_mtx_df.index) - set(spot_idx_to_remove))
    all_count_mtx = all_count_mtx_df.loc[spot_idx_to_keep, :]
    all_img_ebd = all_img_ebd[spot_idx_to_keep, :]
    # normalized by log2(+1)
    all_count_mtx_selected_genes = np.log2(all_count_mtx.loc[:, selected_genes] + 1).copy()

    all_img_ebd.requires_grad_(False)
    train_dataset = CustomDataset(
        torch.from_numpy(all_count_mtx_selected_genes.values).float(),
        all_img_ebd.float(),
    )
    # Build eval dataset from held-out slide(s)
    eval_selected_genes = selected_genes
    eval_count_mtx_list = []
    eval_img_ebd_list = []
    for slide in input_args.slide_out.split(","):
        test_adata = anndata.read_h5ad(input_args.data_path + "st/" + slide + ".h5ad")
        test_count_df = pd.DataFrame(
            test_adata[:, eval_selected_genes].X.toarray(),
            columns=eval_selected_genes,
            index=[slide + "_" + str(i) for i in range(test_adata.shape[0])],
        )
        img_ebd, _ = _load_slide_embeddings(input_args.data_path, slide, input_args.image_encoder, load_aug=False)
        eval_count_mtx_list.append(test_count_df)
        eval_img_ebd_list.append(img_ebd)

    eval_count_mtx_df = pd.concat(eval_count_mtx_list, axis=0)
    # apply same log2(+1) normalization
    eval_count_mtx_selected_genes = np.log2(eval_count_mtx_df + 1).copy()
    eval_img_ebd = torch.cat(eval_img_ebd_list, axis=0).float()

    # Create eval dataset with same format as training dataset
    eval_dataset = CustomDataset(
        torch.from_numpy(eval_count_mtx_selected_genes.values).float(),
        eval_img_ebd,
    )

    return train_dataset, eval_dataset


def load_train_objs(args, logger):
    train_set, eval_data = assemble_dataset(args, logger)
    if args.model.lower() == "stem":
        model = StemModel(
            input_size=args.input_gene_size,
            depth=args.DiT_num_blocks,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            label_size=args.cond_size,
        )
    elif is_bleep_model(args.model):
        model = build_bleep_any_from_omegaconf(args)
    elif args.model.lower() in {"histogene", "histogene_enhanced"}:
        model = build_histogene_from_omegaconf(args)
    elif args.model.lower() == "histogene_original":
        model = build_histogene_original_from_omegaconf(args)
    elif args.model.lower() == "ours":
        if getattr(args, "head_type", "attention") != "attention":
            raise ValueError("The public SimpleP2G entry point uses the gene-query attention decoder")
        model = SimpleP2G(args)
    elif args.model.lower() in CROSS_PATCH_SIMPLE_MODELS:
        model = SimpleP2GNeigh(args)
    elif args.model.lower() == "strank":
        from src.baselines.strank.model import STRankLinear

        model = STRankLinear(args)
    elif args.model.lower() == "stflow":
        from src.baselines.stflow.api import build_stflow_from_omegaconf

        model = build_stflow_from_omegaconf(args)
    elif args.model.lower() == "merge":
        from src.baselines.merge.model import build_merge_from_omegaconf

        model = build_merge_from_omegaconf(args)
    elif args.model.lower() == "m2tglgo":
        from src.baselines.m2tglgo.model import build_m2tglgo_from_omegaconf

        model = build_m2tglgo_from_omegaconf(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    logger.info(f"Train size {len(train_set):,}")
    logger.info(f"Eval size {len(eval_data):,}")
    return train_set, eval_data, model, args

