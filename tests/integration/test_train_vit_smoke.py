import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _single_run_dir(output_root: Path, experiment_name: str) -> Path:
    exp_root = output_root / experiment_name
    run_dirs = [p for p in exp_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"Expected one run dir, found {len(run_dirs)}"
    return run_dirs[0]


def test_train_vit_smoke(
    synthetic_dataset_roots: tuple[Path, Path],
    write_config_factory,
    tiny_binary_model_class,
    monkeypatch: pytest.MonkeyPatch,
    assert_has_metric_keys,
    tmp_path: Path,
) -> None:
    import train_vit as script_mod

    isic_root, ham_root = synthetic_dataset_roots
    output_root = tmp_path / "outputs"
    cfg_path = write_config_factory(
        config_name="vit_smoke.yml",
        experiment_name="vit_smoke",
        output_root=output_root,
        isic_root=isic_root,
        ham_root=ham_root,
        epochs=1,
        batch_size=8,
        num_workers=0,
        image_size=64,
        warmup_epochs=0,
    )

    monkeypatch.setattr(
        script_mod,
        "ViTBinaryClassifier",
        lambda pretrained=True: tiny_binary_model_class(),
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", ["train_vit.py", "--config", str(cfg_path), "--threshold", "0.5"])

    script_mod.main()

    run_dir = _single_run_dir(output_root, "vit_smoke")
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "predictions" / "val_final.csv").exists()
    assert (run_dir / "predictions" / "ham_test.csv").exists()
    assert (run_dir / "metrics" / "curves" / "val_final_pr.png").exists()
    assert (run_dir / "metrics" / "curves" / "ham_test_pr.png").exists()

    val_metrics = json.loads((run_dir / "metrics" / "val_metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((run_dir / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
    assert_has_metric_keys(val_metrics)
    assert_has_metric_keys(test_metrics)
