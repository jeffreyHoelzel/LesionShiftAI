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


def test_smoke_data_pipeline_script(
    synthetic_dataset_roots: tuple[Path, Path],
    write_config_factory,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    import smoke_data_pipeline as script_mod

    isic_root, ham_root = synthetic_dataset_roots
    cfg_path = write_config_factory(
        config_name="smoke_script.yml",
        experiment_name="smoke_script",
        isic_root=isic_root,
        ham_root=ham_root,
        epochs=1,
        num_workers=0,
        image_size=64,
    )

    monkeypatch.setattr(sys, "argv", ["smoke_data_pipeline.py", "--config", str(cfg_path)])
    script_mod.main()

    out = capsys.readouterr().out
    assert "train:" in out
    assert "val:" in out
    assert "test:" in out
