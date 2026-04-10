from dataclasses import dataclass
from typing import Dict
import pandas as pd
from torch.utils.data import DataLoader
from lesionshiftai.core.config import ExperimentConfig
from lesionshiftai.core.reproducibility import init_generator, seed_worker
from lesionshiftai.data.dataset import LesionDataset
from lesionshiftai.data.metadata import load_ham_metadata, load_isic_metadata
from lesionshiftai.data.split import split_isic_train_val
from lesionshiftai.data.transforms import build_eval_transform, build_train_transform


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def build_data_bundle(cfg: ExperimentConfig) -> DataBundle:
    isic_df = load_isic_metadata(cfg.data.isic_root)
    ham_df = load_ham_metadata(cfg.data.ham_root)

    train_df, val_df = split_isic_train_val(
        isic_df=isic_df,
        val_size=cfg.data.val_size,
        seed=cfg.seed
    )

    train_ds = LesionDataset(
        train_df, build_train_transform(cfg.data.image_size)
    )
    eval_tf = build_eval_transform(cfg.data.image_size)
    val_ds = LesionDataset(val_df, eval_tf)
    test_ds = LesionDataset(ham_df, eval_tf)

    common_loader_args = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": cfg.data.num_workers > 0
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=init_generator(cfg.seed),
        **common_loader_args
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        generator=init_generator(cfg.seed + 1),
        **common_loader_args
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        generator=init_generator(cfg.seed + 2),
        **common_loader_args
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_df=train_df,
        val_df=val_df,
        test_df=ham_df
    )


def binary_counts(df: pd.DataFrame) -> Dict[int, int]:
    counts = df["label"].value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}
