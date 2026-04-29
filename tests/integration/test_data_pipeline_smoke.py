import io
import sys
from pathlib import Path

import pytest

from lesionshiftai.core.config import load_config
from lesionshiftai.data.datamodule import build_data_bundle


pytestmark = pytest.mark.integration


def test_build_data_bundle_end_to_end(
    synthetic_dataset_roots: tuple[Path, Path],
    write_config_factory,
) -> None:
    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(
        config_name="smoke_data.yml",
        experiment_name="smoke_data_bundle",
        isic_root=isic_root,
        ham_root=ham_root,
        epochs=1,
        num_workers=0,
        image_size=64,
    )

    cfg = load_config(cfg_path)
    bundle = build_data_bundle(cfg)

    assert len(bundle.train_df) > 0
    assert len(bundle.val_df) > 0
    assert len(bundle.test_df) > 0
    batch = next(iter(bundle.train_loader))
    assert tuple(batch["image"].shape[1:]) == (3, 64, 64)
