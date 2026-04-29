import importlib
import json
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.unit


def _seed_member_artifacts(ensemble_root: Path, num_folds: int) -> None:
    for fold in range(num_folds):
        member_dir = ensemble_root / "members" / f"fold_{fold}"
        (member_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (member_dir / "metrics" / "curves").mkdir(parents=True, exist_ok=True)

        val_df = pd.DataFrame(
            {
                "sample_id": [f"isic_{fold}_0", f"isic_{fold}_1"],
                "dataset": ["isic2019", "isic2019"],
                "label": [0, 1],
                "prob_malignant": [0.2 + 0.05 * fold, 0.7 - 0.05 * fold],
                "pred_label": [0, 1],
            }
        )
        val_df.to_csv(member_dir / "predictions" / "val_final.csv", index=False)

        ham_df = pd.DataFrame(
            {
                "sample_id": ["ham_0", "ham_1", "ham_2", "ham_3"],
                "dataset": ["ham10000"] * 4,
                "label": [0, 1, 0, 1],
                "prob_malignant": [0.2 + 0.02 * fold, 0.8 - 0.01 * fold, 0.3, 0.7],
                "pred_label": [0, 1, 0, 1],
            }
        )
        ham_df.to_csv(member_dir / "predictions" / "ham_test.csv", index=False)

        val_metrics = {
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1": 0.8,
            "roc_auc": 0.85,
            "pr_auc": 0.7,
            "tn": 1,
            "fp": 0,
            "fn": 0,
            "tp": 1,
            "loss": 0.4,
        }
        test_metrics = {
            "accuracy": 0.78,
            "precision": 0.75,
            "recall": 0.77,
            "f1": 0.76,
            "roc_auc": 0.82,
            "pr_auc": 0.65,
            "tn": 2,
            "fp": 0,
            "fn": 1,
            "tp": 1,
            "loss": 0.5,
        }
        (member_dir / "metrics" / "val_metrics.json").write_text(
            json.dumps(val_metrics), encoding="utf-8"
        )
        (member_dir / "metrics" / "test_metrics.json").write_text(
            json.dumps(test_metrics), encoding="utf-8"
        )
        (member_dir / "metrics" / "member_complete.json").write_text(
            json.dumps({"status": "complete", "fold_index": fold}), encoding="utf-8"
        )
        (member_dir / "metrics" / "history.json").write_text(
            json.dumps(
                {
                    "epochs": [
                        {
                            "epoch": 1,
                            "train": {"loss": 0.6},
                            "val": {"roc_auc": 0.72, "pr_auc": 0.58},
                        },
                        {
                            "epoch": 2,
                            "train": {"loss": 0.5},
                            "val": {"roc_auc": 0.8, "pr_auc": 0.66},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        val_curve = {
            "status": "ok",
            "fold_index": fold,
            "roc_auc": 0.85,
            "pr_auc": 0.7,
            "n_samples": 2,
            "n_positive": 1,
            "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.9, 1.0]},
            "pr_curve": {"recall": [1.0, 0.5, 0.0], "precision": [0.5, 0.9, 1.0]},
        }
        ham_curve = {
            "status": "ok",
            "fold_index": fold,
            "roc_auc": 0.82,
            "pr_auc": 0.65,
            "n_samples": 4,
            "n_positive": 2,
            "roc_curve": {"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0]},
            "pr_curve": {"recall": [1.0, 0.7, 0.0], "precision": [0.5, 0.85, 1.0]},
        }
        (member_dir / "metrics" / "curves" / "val_final_curves.json").write_text(
            json.dumps(val_curve), encoding="utf-8"
        )
        (member_dir / "metrics" / "curves" / "ham_test_curves.json").write_text(
            json.dumps(ham_curve), encoding="utf-8"
        )


def test_ensemble_path_helpers() -> None:
    mod = importlib.import_module("train_ensemble_member_cnn")
    root = mod._ensemble_root(Path("out"), "exp", "run1")
    assert root.as_posix() == "out/exp/ensemble_run1"
    assert mod._member_dir_from_root(root, 3).as_posix().endswith("members/fold_3")


def test_write_ensemble_validation_pending(tmp_path: Path) -> None:
    mod = importlib.import_module("train_ensemble_member_cnn")
    ensemble_root = tmp_path / "out" / "exp" / "ensemble_runx"
    (ensemble_root / "members" / "fold_0" / "metrics").mkdir(parents=True, exist_ok=True)
    (ensemble_root / "members" / "fold_0" / "metrics" / "member_complete.json").write_text("{}", encoding="utf-8")

    status = mod._write_ensemble_validation_if_ready(
        ensemble_root=ensemble_root,
        ensemble_run_id="runx",
        num_folds=2,
        threshold=0.5,
    )
    assert status["status"] == "pending"
    assert status["missing_folds"] == [1]


def test_write_ensemble_validation_completed(tmp_path: Path) -> None:
    mod = importlib.import_module("train_ensemble_member_cnn")
    ensemble_root = tmp_path / "out" / "exp" / "ensemble_runy"
    _seed_member_artifacts(ensemble_root, num_folds=2)

    status = mod._write_ensemble_validation_if_ready(
        ensemble_root=ensemble_root,
        ensemble_run_id="runy",
        num_folds=2,
        threshold=0.5,
    )

    assert status["status"] == "completed"
    curves_dir = ensemble_root / "ensemble" / "metrics" / "curves"
    assert (curves_dir / "isic_val_aggregate_curves.json").exists()
    assert (curves_dir / "isic_val_member_folds_curves.json").exists()
    assert (curves_dir / "isic_val_member_folds_auc_history.json").exists()
    assert (curves_dir / "isic_val_member_folds_roc_auc_history.png").exists()
    assert (curves_dir / "isic_val_member_folds_pr_auc_history.png").exists()
    assert (curves_dir / "ham_test_aggregate_curves.json").exists()
    assert (curves_dir / "ham_test_member_folds_curves.json").exists()
    assert (curves_dir / "ham_test_member_folds_auc_history.json").exists()
    assert (curves_dir / "ham_test_member_folds_roc_auc_history.png").exists()
    assert (curves_dir / "ham_test_member_folds_pr_auc_history.png").exists()

    metrics_dir = ensemble_root / "ensemble" / "metrics"
    assert (metrics_dir / "isic_val_aggregate_metrics.json").exists()
    assert (metrics_dir / "ham_test_aggregate_metrics.json").exists()
    assert (metrics_dir / "generalization_gap.json").exists()
