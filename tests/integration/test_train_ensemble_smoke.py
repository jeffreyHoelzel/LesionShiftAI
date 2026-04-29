import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_train_ensemble_smoke(
    synthetic_dataset_factory,
    write_config_factory,
    tiny_binary_model_class,
    monkeypatch: pytest.MonkeyPatch,
    assert_has_metric_keys,
    tmp_path: Path,
) -> None:
    import train_ensemble_member_cnn as script_mod

    isic_root, ham_root = synthetic_dataset_factory(n_isic=80, n_ham=24, grouped=True)
    output_root = tmp_path / "outputs"
    cfg_path = write_config_factory(
        config_name="ensemble_smoke.yml",
        experiment_name="ensemble_smoke",
        output_root=output_root,
        isic_root=isic_root,
        ham_root=ham_root,
        epochs=1,
        batch_size=8,
        num_workers=0,
        image_size=64,
    )

    run_id = "smoke_ens"
    monkeypatch.setattr(script_mod, "BaselineCNN", lambda pretrained=True: tiny_binary_model_class())
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ensemble_member_cnn.py",
            "--config",
            str(cfg_path),
            "--num-folds",
            "2",
            "--ensemble-run-id",
            run_id,
            "--threshold",
            "0.5",
        ],
    )

    script_mod.main()

    ensemble_root = output_root / "ensemble_smoke" / f"ensemble_{run_id}"
    for fold_idx in (0, 1):
        member_dir = ensemble_root / "members" / f"fold_{fold_idx}"
        assert (member_dir / "checkpoints" / "best.pt").exists()
        assert (member_dir / "predictions" / "val_final.csv").exists()
        assert (member_dir / "predictions" / "ham_test.csv").exists()
        val_metrics = json.loads((member_dir / "metrics" / "val_metrics.json").read_text(encoding="utf-8"))
        test_metrics = json.loads((member_dir / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
        assert_has_metric_keys(val_metrics)
        assert_has_metric_keys(test_metrics)

    agg_metrics_dir = ensemble_root / "ensemble" / "metrics"
    agg_preds_dir = ensemble_root / "ensemble" / "predictions"
    agg_curves_dir = agg_metrics_dir / "curves"

    assert (agg_preds_dir / "isic_val_aggregate_predictions.csv").exists()
    assert (agg_preds_dir / "ham_test_aggregate_predictions.csv").exists()
    assert (agg_metrics_dir / "isic_val_aggregate_metrics.json").exists()
    assert (agg_metrics_dir / "ham_test_aggregate_metrics.json").exists()
    assert (agg_metrics_dir / "generalization_gap.json").exists()

    assert (agg_curves_dir / "isic_val_aggregate_curves.json").exists()
    assert (agg_curves_dir / "isic_val_member_folds_curves.json").exists()
    assert (agg_curves_dir / "isic_val_member_folds_auc_history.json").exists()
    assert (agg_curves_dir / "isic_val_member_folds_roc_auc_history.png").exists()
    assert (agg_curves_dir / "isic_val_member_folds_pr_auc_history.png").exists()
    assert (agg_curves_dir / "ham_test_aggregate_curves.json").exists()
    assert (agg_curves_dir / "ham_test_member_folds_curves.json").exists()
    assert (agg_curves_dir / "ham_test_member_folds_auc_history.json").exists()
    assert (agg_curves_dir / "ham_test_member_folds_roc_auc_history.png").exists()
    assert (agg_curves_dir / "ham_test_member_folds_pr_auc_history.png").exists()
