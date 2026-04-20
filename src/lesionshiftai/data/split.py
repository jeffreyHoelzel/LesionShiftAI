from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def split_isic_train_val(
    isic_df: pd.DataFrame,
    val_size: float = 0.20,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_size < 0.5:
        raise ValueError("`val_size` must be between 0 and 0.5")

    # split by patient ID groupings, otherwise use normal train-test split
    has_true_groups = isic_df["patient_id"].nunique() < len(isic_df)
    if has_true_groups:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=val_size, random_state=seed)
        train_index, val_index = next(
            splitter.split(
                isic_df, y=isic_df["label"], groups=isic_df["patient_id"])
        )
        train_df = isic_df.iloc[train_index].copy()
        val_df = isic_df.iloc[val_index].copy()
    else:
        train_index, val_index = train_test_split(
            isic_df.index.to_numpy(),
            test_size=val_size,
            stratify=isic_df["label"],
            random_state=seed,
            shuffle=True
        )
        train_df = isic_df.loc[train_index].copy()
        val_df = isic_df.loc[val_index].copy()

    # should be no overlap
    overlap = set(train_df["sample_id"]).intersection(val_df["sample_id"])
    if overlap:
        raise RuntimeError("Leakage detected: train/val sample overlaps")

    # should always be malignant (1) or benign (0)
    if train_df["label"].nunique() < 2 or val_df["label"].nunique() < 2:
        raise RuntimeError("Split failed: one split is missing a class")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def assign_isic_folds(
    isic_df: pd.DataFrame,
    num_folds: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    if num_folds < 2:
        raise ValueError("`num_folds` must be >= 2")
    if len(isic_df) < num_folds:
        raise ValueError("`num_folds` cannot exceed number of ISIC rows")
    if isic_df["label"].nunique() < 2:
        raise RuntimeError(
            "Fold assignment failed: ISIC metadata has one class")

    has_true_groups = isic_df["patient_id"].nunique() < len(isic_df)
    fold_ids = np.full(shape=len(isic_df), fill_value=-1, dtype=np.int64)

    if has_true_groups:
        splitter = StratifiedGroupKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed
        )
        split_iter = splitter.split(
            X=isic_df,
            y=isic_df["label"],
            groups=isic_df["patient_id"]
        )
    else:
        splitter = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=seed
        )
        split_iter = splitter.split(X=isic_df, y=isic_df["label"])

    for fold_index, (_, holdout_index) in enumerate(split_iter):
        fold_ids[holdout_index] = fold_index

    out = isic_df.copy().reset_index(drop=True)
    out["fold"] = fold_ids.astype(int)

    _validate_fold_assignment(
        fold_df=out,
        num_folds=num_folds,
        grouped_by_patient=has_true_groups
    )
    return out


def summarize_fold_assignment(
    fold_df: pd.DataFrame,
    num_folds: int
) -> Dict[str, Any]:
    fold_counts = fold_df["fold"].value_counts().sort_index()
    label_counts = (
        fold_df.groupby(["fold", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    folds: Dict[str, Dict[str, int]] = {}
    for fold_idx in range(num_folds):
        benign = int(label_counts.loc[fold_idx, 0]
                     ) if 0 in label_counts.columns else 0
        malignant = (
            int(label_counts.loc[fold_idx, 1])
            if 1 in label_counts.columns
            else 0
        )
        folds[str(fold_idx)] = {
            "n_samples": int(fold_counts.get(fold_idx, 0)),
            "n_benign": benign,
            "n_malignant": malignant
        }

    return {
        "num_folds": int(num_folds),
        "n_samples_total": int(len(fold_df)),
        "folds": folds
    }


def _validate_fold_assignment(
    fold_df: pd.DataFrame,
    num_folds: int,
    grouped_by_patient: bool
) -> None:
    if (fold_df["fold"] < 0).any():
        raise RuntimeError("Fold assignment failed: some rows were unassigned")

    seen_folds = set(fold_df["fold"].unique().tolist())
    expected_folds = set(range(num_folds))
    if seen_folds != expected_folds:
        raise RuntimeError(
            f"Fold assignment failed: expected folds {sorted(expected_folds)}, "
            f"got {sorted(seen_folds)}"
        )

    counts = fold_df["fold"].value_counts().sort_index()
    if (counts <= 0).any():
        raise RuntimeError(
            "Fold assignment failed: one or more folds are empty")

    expected = len(fold_df) / float(num_folds)
    max_dev = float(np.abs(counts.to_numpy(dtype=float) - expected).max())
    if max_dev > expected * 0.60:
        raise RuntimeError(
            "Fold assignment failed: fold sizes are too imbalanced "
            f"(max deviation {max_dev:.1f}, expected {expected:.1f})"
        )

    per_fold_class_count = fold_df.groupby("fold")["label"].nunique()
    if (per_fold_class_count < 2).any():
        bad_folds = per_fold_class_count[per_fold_class_count < 2].index.tolist(
        )
        raise RuntimeError(
            f"Fold assignment failed: missing class in folds {bad_folds}"
        )

    if grouped_by_patient:
        patient_fold_counts = fold_df.groupby("patient_id")["fold"].nunique()
        leaked_patients = patient_fold_counts[patient_fold_counts > 1]
        if not leaked_patients.empty:
            preview = leaked_patients.index.astype(str).tolist()[:5]
            raise RuntimeError(
                "Leakage detected: patient IDs appear in multiple folds. "
                f"Examples: {preview}"
            )
