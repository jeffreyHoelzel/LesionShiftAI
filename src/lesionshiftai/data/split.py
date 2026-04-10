from typing import Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


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
