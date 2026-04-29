import json
from pathlib import Path

import pytest

from lesionshiftai.eval.curves import (
    write_binary_curve_artifacts,
    write_fold_auc_history_overlay_artifacts,
    write_fold_curve_overlay_artifacts,
)


pytestmark = pytest.mark.unit


def test_write_binary_curve_artifacts_success(tmp_path: Path) -> None:
    y_true = [0, 0, 1, 1, 0, 1]
    y_prob = [0.1, 0.3, 0.8, 0.9, 0.2, 0.7]

    payload = write_binary_curve_artifacts(
        y_true=y_true,
        y_prob=y_prob,
        output_dir=tmp_path,
        split_name="val_final",
        model_scope="baseline",
        extra_metadata={"threshold": 0.5},
    )

    assert payload["status"] == "ok"
    assert (tmp_path / "val_final_curves.json").exists()
    assert (tmp_path / "val_final_roc.png").exists()
    assert (tmp_path / "val_final_pr.png").exists()


def test_write_binary_curve_artifacts_single_class_skips(tmp_path: Path) -> None:
    payload = write_binary_curve_artifacts(
        y_true=[1, 1, 1],
        y_prob=[0.5, 0.6, 0.7],
        output_dir=tmp_path,
        split_name="ham_test",
        model_scope="baseline",
    )

    assert payload["status"] == "skipped_single_class"
    assert (tmp_path / "ham_test_curves.json").exists()
    assert not (tmp_path / "ham_test_roc.png").exists()


def test_write_fold_curve_overlay_artifacts_success(tmp_path: Path) -> None:
    fold_payloads = [
        {
            "status": "ok",
            "fold_index": 0,
            "roc_auc": 0.8,
            "pr_auc": 0.7,
            "n_samples": 10,
            "n_positive": 4,
            "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.8, 1.0]},
            "pr_curve": {"recall": [1.0, 0.5, 0.0], "precision": [0.4, 0.8, 1.0]},
        },
        {
            "status": "ok",
            "fold_index": 1,
            "roc_auc": 0.75,
            "pr_auc": 0.65,
            "n_samples": 12,
            "n_positive": 5,
            "roc_curve": {"fpr": [0.0, 0.4, 1.0], "tpr": [0.0, 0.7, 1.0]},
            "pr_curve": {"recall": [1.0, 0.6, 0.0], "precision": [0.42, 0.78, 1.0]},
        },
    ]

    payload = write_fold_curve_overlay_artifacts(
        fold_curve_payloads=fold_payloads,
        output_dir=tmp_path,
        split_name="isic_val_member_folds",
        model_scope="ensemble_member_folds",
    )

    assert payload["status"] == "ok"
    assert payload["n_folds_plotted"] == 2
    assert (tmp_path / "isic_val_member_folds_curves.json").exists()
    assert (tmp_path / "isic_val_member_folds_roc.png").exists()
    assert (tmp_path / "isic_val_member_folds_pr.png").exists()


def test_write_fold_curve_overlay_artifacts_handles_no_valid_folds(tmp_path: Path) -> None:
    payload = write_fold_curve_overlay_artifacts(
        fold_curve_payloads=[{"status": "skipped_single_class", "fold_index": 0}],
        output_dir=tmp_path,
        split_name="ham_test_member_folds",
        model_scope="ensemble_member_folds",
    )

    assert payload["status"] == "skipped_no_valid_folds"
    parsed = json.loads((tmp_path / "ham_test_member_folds_curves.json").read_text(encoding="utf-8"))
    assert parsed["reason"] == "no_valid_fold_curve_payloads"


def test_write_fold_auc_history_overlay_artifacts_success(tmp_path: Path) -> None:
    fold_history_payloads = [
        {
            "fold_index": 0,
            "epochs": [
                {"epoch": 1, "val": {"roc_auc": 0.72, "pr_auc": 0.55}},
                {"epoch": 2, "val": {"roc_auc": 0.78, "pr_auc": 0.61}},
            ],
        },
        {
            "fold_index": 1,
            "epochs": [
                {"epoch": 1, "val": {"roc_auc": 0.7, "pr_auc": 0.52}},
                {"epoch": 2, "val": {"roc_auc": 0.75, "pr_auc": 0.6}},
            ],
        },
    ]

    payload = write_fold_auc_history_overlay_artifacts(
        fold_history_payloads=fold_history_payloads,
        output_dir=tmp_path,
        split_name="isic_val_member_folds",
        model_scope="ensemble_member_folds",
    )

    assert payload["status"] == "ok"
    assert payload["n_folds_plotted"] == 2
    assert (tmp_path / "isic_val_member_folds_auc_history.json").exists()
    assert (tmp_path / "isic_val_member_folds_roc_auc_history.png").exists()
    assert (tmp_path / "isic_val_member_folds_pr_auc_history.png").exists()


def test_write_fold_auc_history_overlay_artifacts_handles_no_valid_folds(
    tmp_path: Path,
) -> None:
    payload = write_fold_auc_history_overlay_artifacts(
        fold_history_payloads=[{"fold_index": 0, "epochs": []}],
        output_dir=tmp_path,
        split_name="isic_val_member_folds",
        model_scope="ensemble_member_folds",
    )

    assert payload["status"] == "skipped_no_valid_folds"
    parsed = json.loads(
        (tmp_path / "isic_val_member_folds_auc_history.json").read_text(
            encoding="utf-8"
        )
    )
    assert parsed["reason"] == "no_valid_fold_auc_histories"
