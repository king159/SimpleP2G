"""Target-patch features with raw-count/slide-ID supervision for STRank."""

import anndata
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class STRankSpotDataset(Dataset):
    def __init__(self, counts, features, sample_ids):
        self.count = torch.as_tensor(counts, dtype=torch.float32)
        self.features = features.float().detach()
        self.sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)
        self.labels = torch.from_numpy(np.log2(np.asarray(counts) + 1)).float()
        if not (len(self.count) == len(self.features) == len(self.sample_ids)):
            raise ValueError("STRank features, counts and slide IDs are misaligned")
        if not torch.isfinite(self.count).all() or (self.count < 0).any():
            raise ValueError("STRank requires finite nonnegative raw counts")
        if not torch.isfinite(self.features).all():
            raise ValueError("Non-finite STRank image features")

    def __len__(self):
        return len(self.count)

    def __getitem__(self, index):
        return {"img_feat": self.features[index], "count": self.count[index],
                "sample_id": self.sample_ids[index], "labels": self.labels[index]}


def assemble_strank_datasets(args, logger, train_slides, genes, load_embeddings):
    test_slides = args.slide_out.split(",")
    assert set(train_slides).isdisjoint(test_slides)
    slide_map = {slide: index for index, slide in enumerate([*train_slides, *test_slides])}

    def load(slides, training):
        counts, features, augmented, ids = [], [], [], []
        for slide in slides:
            adata = anndata.read_h5ad(args.data_path + "st/" + slide + ".h5ad")
            matrix = adata[:, genes].X
            matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
            feature, augmentation = load_embeddings(args.data_path, slide, args.image_encoder, load_aug=training)
            assert len(matrix) == len(feature)
            counts.append(matrix)
            features.append(feature)
            ids.append(np.full(len(matrix), slide_map[slide], dtype=np.int64))
            if training:
                augmented.append(augmentation)
        count = np.concatenate(counts)
        feature = torch.cat(features)
        sample_id = np.concatenate(ids)
        if training:
            augmentation = torch.cat(augmented)
            ratio = int(args.num_aug_ratio)
            selected = torch.zeros((len(count) * ratio, feature.shape[1]))
            for index in range(len(count)):
                choices = np.random.choice(augmentation.shape[1], ratio, replace=False)
                selected[index*ratio:(index+1)*ratio] = augmentation[index, choices]
            feature = torch.cat([feature, selected])
            count = np.concatenate([count, np.repeat(count, ratio, axis=0)])
            sample_id = np.concatenate([sample_id, np.repeat(sample_id, ratio)])
            # Same training-only all-zero filtering as the shared spot loader.
            keep = ~(np.isnan(count).all(axis=1) | (np.nansum(count, axis=1) == 0))
            count, feature, sample_id = count[keep], feature[keep], sample_id[keep]
        return STRankSpotDataset(count, feature, sample_id)

    train = load(train_slides, True)
    test = load(test_slides, False)
    logger.info("STRank official objective: %d training items, %d training slides, %d test spots; test forward uses target image only",
                len(train), len(train_slides), len(test))
    return train, test
