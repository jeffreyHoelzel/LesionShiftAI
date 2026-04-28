import numpy as np
import pandas as pd
import pytest

from lesionshiftai.data.split import (
    _validate_fold_assignment,
    assign_isic_folds,
    split_isic_train_val,
    summarize_fold_assignment,
)


pytestmark = pytest.mark.unit


def _make_isic_df(n: int = 40, grouped: bool = True) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "sample_id": f"S{idx:04d}",
                "patient_id": f"P{idx // 2:03d}" if grouped else f"P{idx:03d}",
                "image_path": f"/tmp/{idx}.jpg",
                "label": idx % 2,
                "source_class": "malignant" if idx % 2 else "benign",
                "dataset": "isic2019",
            }
        )
    return pd.DataFrame(rows)


def test_split_isic_train_val_success() -> None:
    df = _make_isic_df(n=60, grouped=True)
    train_df, val_df = split_isic_train_val(df, val_size=0.2, seed=1)

    assert len(train_df) + len(val_df) == len(df)
    assert not set(train_df["sample_id"]).intersection(set(val_df["sample_id"]))
    assert train_df["label"].nunique() == 2
    assert val_df["label"].nunique() == 2


@pytest.mark.parametrize("val_size", [0.0, 0.5, -0.1])
def test_split_isic_train_val_rejects_invalid_size(val_size: float) -> None:
    with pytest.raises(ValueError):
        split_isic_train_val(_make_isic_df(), val_size=val_size)


def test_split_isic_train_val_rejects_missing_class() -> None:
    df = _make_isic_df(n=20)
    df["label"] = 0
    with pytest.raises(RuntimeError):
        split_isic_train_val(df, val_size=0.2, seed=42)


def test_assign_isic_folds_success_grouped() -> None:
    df = _make_isic_df(n=60, grouped=True)
    fold_df = assign_isic_folds(df, num_folds=5, seed=9)

    assert "fold" in fold_df.columns
    assert set(fold_df["fold"].unique()) == {0, 1, 2, 3, 4}

    # grouped split should not leak patient IDs across folds
    patient_counts = fold_df.groupby("patient_id")["fold"].nunique()
    assert int(patient_counts.max()) == 1


@pytest.mark.parametrize(
    ("num_folds", "n_rows"),
    [(1, 20), (10, 5)],
)
def test_assign_isic_folds_rejects_invalid_fold_configuration(num_folds: int, n_rows: int) -> None:
    with pytest.raises(ValueError):
        assign_isic_folds(_make_isic_df(n=n_rows), num_folds=num_folds)


def test_assign_isic_folds_rejects_single_class() -> None:
    df = _make_isic_df(n=20)
    df["label"] = 1
    with pytest.raises(RuntimeError):
        assign_isic_folds(df, num_folds=2)


def test_summarize_fold_assignment_shape() -> None:
    df = _make_isic_df(n=30)
    fold_df = assign_isic_folds(df, num_folds=3, seed=7)
    summary = summarize_fold_assignment(fold_df, num_folds=3)

    assert summary["num_folds"] == 3
    assert summary["n_samples_total"] == 30
    assert set(summary["folds"].keys()) == {"0", "1", "2"}


def test_validate_fold_assignment_rejects_unassigned_rows() -> None:
    fold_df = _make_isic_df(n=20)
    fold_df["fold"] = np.where(fold_df.index == 0, -1, fold_df.index % 2)
    with pytest.raises(RuntimeError):
        _validate_fold_assignment(fold_df, num_folds=2, grouped_by_patient=True)
