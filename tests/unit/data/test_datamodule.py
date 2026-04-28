from pathlib import Path

import pytest
import torch

from lesionshiftai.core.config import load_config
from lesionshiftai.data.datamodule import (
    _build_train_val_loaders,
    _common_loader_args,
    binary_counts,
    build_data_bundle,
    build_isic_fold_data_bundle,
)
from lesionshiftai.data.dataset import LesionDataset
from lesionshiftai.data.transforms import build_eval_transform


pytestmark = pytest.mark.unit


def test_binary_counts() -> None:
    import pandas as pd

    df = pd.DataFrame({"label": [0, 1, 1, 0, 1]})
    assert binary_counts(df) == {0: 2, 1: 3}


def test_common_loader_args(synthetic_dataset_roots: tuple[Path, Path], write_config_factory) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(isic_root=isic_root, ham_root=ham_root)
    cfg = load_config(cfg_path)
    args = _common_loader_args(cfg)

    assert args["batch_size"] == cfg.data.batch_size
    assert args["num_workers"] == cfg.data.num_workers
    assert "worker_init_fn" in args


def test_build_data_bundle_single_process(synthetic_dataset_roots: tuple[Path, Path], write_config_factory) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(isic_root=isic_root, ham_root=ham_root, num_workers=0)
    cfg = load_config(cfg_path)

    bundle = build_data_bundle(cfg, world_size=1, rank=0)

    assert len(bundle.train_df) > 0
    assert len(bundle.val_df) > 0
    assert len(bundle.test_df) > 0
    assert bundle.train_sampler is None
    assert bundle.val_sampler is None
    assert bundle.test_sampler is None


def test_build_data_bundle_distributed_samplers(synthetic_dataset_roots: tuple[Path, Path], write_config_factory) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(isic_root=isic_root, ham_root=ham_root, num_workers=0)
    cfg = load_config(cfg_path)

    bundle = build_data_bundle(cfg, world_size=2, rank=0)

    assert bundle.train_sampler is not None
    assert bundle.val_sampler is not None
    assert bundle.test_sampler is not None


def test_build_isic_fold_data_bundle_success(synthetic_dataset_roots: tuple[Path, Path], write_config_factory) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(isic_root=isic_root, ham_root=ham_root, num_workers=0)
    cfg = load_config(cfg_path)

    bundle = build_isic_fold_data_bundle(
        cfg=cfg,
        num_folds=4,
        fold_index=1,
        world_size=1,
        rank=0,
    )

    assert len(bundle.train_df) > 0
    assert len(bundle.val_df) > 0
    assert "fold" in bundle.fold_assignment_df.columns


def test_build_isic_fold_data_bundle_invalid_fold_index(
    synthetic_dataset_roots: tuple[Path, Path], write_config_factory
) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(isic_root=isic_root, ham_root=ham_root)
    cfg = load_config(cfg_path)

    with pytest.raises(ValueError):
        build_isic_fold_data_bundle(cfg=cfg, num_folds=3, fold_index=3)


def test_build_train_val_loaders_distributed(
    synthetic_dataset_roots: tuple[Path, Path],
    write_config_factory,
) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(isic_root=isic_root, ham_root=ham_root, num_workers=0)
    cfg = load_config(cfg_path)

    import pandas as pd

    rows = []
    for idx in range(10):
        rows.append(
            {
                "sample_id": f"S{idx:02d}",
                "patient_id": f"P{idx:02d}",
                "image_path": str(isic_root / "train images" / f"ISIC_{idx:05d}.jpg"),
                "label": idx % 2,
                "source_class": "malignant" if idx % 2 else "benign",
                "dataset": "isic2019",
            }
        )
    df = pd.DataFrame(rows)
    ds = LesionDataset(df, build_eval_transform(32))

    train_loader, val_loader, train_sampler, val_sampler = _build_train_val_loaders(
        cfg=cfg,
        train_ds=ds,
        val_ds=ds,
        world_size=2,
        rank=0,
        seed_base=99,
    )

    assert train_sampler is not None
    assert val_sampler is not None
    first_batch = next(iter(train_loader))
    assert "image" in first_batch
    assert "label" in first_batch
    assert isinstance(first_batch["label"], torch.Tensor)
    _ = next(iter(val_loader))
