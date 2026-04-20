from dataclasses import dataclass
from typing import Dict
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lesionshiftai.core.config import ExperimentConfig
from lesionshiftai.core.reproducibility import init_generator, seed_worker
from lesionshiftai.data.dataset import LesionDataset
from lesionshiftai.data.metadata import load_ham_metadata, load_isic_metadata
from lesionshiftai.data.split import assign_isic_folds, split_isic_train_val
from lesionshiftai.data.transforms import build_eval_transform, build_train_transform


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_sampler: DistributedSampler | None
    val_sampler: DistributedSampler | None
    test_sampler: DistributedSampler | None


@dataclass(slots=True)
class IsicFoldDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    fold_assignment_df: pd.DataFrame
    train_sampler: DistributedSampler | None
    val_sampler: DistributedSampler | None


def build_data_bundle(
    cfg: ExperimentConfig,
    world_size: int = 1,
    rank: int = 0
) -> DataBundle:
    isic_df = load_isic_metadata(cfg.data.isic_root)
    ham_df = load_ham_metadata(cfg.data.ham_root)

    train_df, val_df = split_isic_train_val(
        isic_df=isic_df,
        val_size=cfg.data.val_size,
        seed=cfg.seed
    )

    train_ds = LesionDataset(
        train_df, build_train_transform(cfg.data.image_size))
    eval_tf = build_eval_transform(cfg.data.image_size)
    val_ds = LesionDataset(val_df, eval_tf)
    test_ds = LesionDataset(ham_df, eval_tf)

    train_loader, val_loader, train_sampler, val_sampler = _build_train_val_loaders(
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        world_size=world_size,
        rank=rank,
        seed_base=cfg.seed
    )

    common_loader_args = _common_loader_args(cfg)
    test_sampler = None
    if world_size > 1:
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=cfg.seed
        )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        sampler=test_sampler,
        generator=init_generator(cfg.seed + 200 + rank),
        **common_loader_args
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_df=train_df,
        val_df=val_df,
        test_df=ham_df,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        test_sampler=test_sampler
    )


def build_isic_fold_data_bundle(
    cfg: ExperimentConfig,
    num_folds: int = 5,
    fold_index: int = 0,
    world_size: int = 1,
    rank: int = 0
) -> IsicFoldDataBundle:
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError("`fold_index` must be between 0 and num_folds - 1")

    isic_df = load_isic_metadata(cfg.data.isic_root)
    fold_df = assign_isic_folds(isic_df, num_folds=num_folds, seed=cfg.seed)

    member_df = (
        fold_df[fold_df["fold"] == fold_index]
        .drop(columns=["fold"])
        .reset_index(drop=True)
    )
    if member_df.empty:
        raise RuntimeError(f"Fold {fold_index} has no assigned rows")

    train_df, val_df = split_isic_train_val(
        isic_df=member_df,
        val_size=cfg.data.val_size,
        seed=cfg.seed + fold_index
    )

    train_ds = LesionDataset(
        train_df, build_train_transform(cfg.data.image_size))
    eval_tf = build_eval_transform(cfg.data.image_size)
    val_ds = LesionDataset(val_df, eval_tf)

    seed_base = cfg.seed + (fold_index * 1000)
    train_loader, val_loader, train_sampler, val_sampler = _build_train_val_loaders(
        cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        world_size=world_size,
        rank=rank,
        seed_base=seed_base
    )

    return IsicFoldDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_df=train_df,
        val_df=val_df,
        fold_assignment_df=fold_df,
        train_sampler=train_sampler,
        val_sampler=val_sampler
    )


def binary_counts(df: pd.DataFrame) -> Dict[int, int]:
    counts = df["label"].value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


def _common_loader_args(cfg: ExperimentConfig) -> Dict[str, object]:
    return {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": cfg.data.num_workers > 0
    }


def _build_train_val_loaders(
    cfg: ExperimentConfig,
    train_ds: LesionDataset,
    val_ds: LesionDataset,
    world_size: int,
    rank: int,
    seed_base: int
) -> tuple[
    DataLoader,
    DataLoader,
    DistributedSampler | None,
    DistributedSampler | None
]:
    common_loader_args = _common_loader_args(cfg)

    # handle multiple GPUs for training and validation
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed_base
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed_base
        )

    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=init_generator(seed_base + rank),
        **common_loader_args
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        generator=init_generator(seed_base + 100 + rank),
        **common_loader_args
    )
    return train_loader, val_loader, train_sampler, val_sampler
