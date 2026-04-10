from pathlib import Path
import pandas as pd
from lesionshiftai.data.labels import (
    BENIGN,
    MALIGNANT,
    HAM_CLASS_COLUMNS,
    HAM_MALIGNANT_CLASSES
)


def load_isic_metadata(
    isic_root: str | Path,
    strict_images: bool = True
) -> pd.DataFrame:
    isic_root = Path(isic_root)
    csv_path = isic_root / "train-metadata.csv"
    image_dir = isic_root / "train images"

    df = pd.read_csv(csv_path)
    if "Unnamed 0" in df.columns:
        df = df.drop(columns=["Unnamed 0"])

    required = {"isic_id", "patient_id", "target"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ISIC missing columns: {sorted(missing)}")

    # ensure correct types and replace empty patient ID with dup sample ID
    df["sample_id"] = df["isic_id"].astype(str)
    df["patient_id"] = df["patient_id"].fillna(df["sample_id"]).astype(str)
    df["label"] = df["target"].astype(int)

    # verify no invalid labels != {0, 1}
    bad_labels = sorted(set(df["label"].unique()) - {BENIGN, MALIGNANT})
    if bad_labels:
        raise ValueError(f"ISIC has non-binary labels: {bad_labels}")

    df["source_class"] = df["label"].map(
        {BENIGN: "benign", MALIGNANT: "malignant"})
    df["dataset"] = "isic2019"
    # map each sample ID to actual image path
    df["image_path"] = df["sample_id"].map(
        lambda sid: str(image_dir / f"{sid}.jpg"))

    out = df[["sample_id", "patient_id", "image_path",
              "label", "source_class", "dataset"]]

    # ensure each image actually exists per path
    if strict_images:
        _assert_paths_exist(out["image_path"], "ISIC 2019")

    return out.reset_index(drop=True)


def load_ham_metadata(ham_root: str | Path, strict_images: bool = True) -> pd.DataFrame:
    ham_root = Path(ham_root)
    csv_path = ham_root / "GroundTruth.csv"
    image_dir = ham_root / "images"

    df = pd.read_csv(csv_path).copy()

    required = {"image", *HAM_CLASS_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"HAM10000 missing columns: {sorted(missing)}")

    one_hot_sum = df[HAM_CLASS_COLUMNS].sum(axis=1)
    if not (one_hot_sum == 1).all():
        bad = int((one_hot_sum != 1).sum())
        raise ValueError(f"HAM10000 has {bad} invalid one-hot rows")

    # ensure correct types and valid labels
    df["sample_id"] = df["image"].astype(str)
    df["source_class"] = df[HAM_CLASS_COLUMNS].idxmax(axis=1)
    df["label"] = df["source_class"].isin(HAM_MALIGNANT_CLASSES).astype(int)
    # fallback: no patient metadata in this CSV
    df["patient_id"] = df["sample_id"]
    df["dataset"] = "ham10000"
    df["image_path"] = df["sample_id"].map(
        lambda sid: str(image_dir / f"{sid}.jpg"))

    out = df[["sample_id", "patient_id", "image_path",
              "label", "source_class", "dataset"]]

    # ensure each image actually exists per path
    if strict_images:
        _assert_paths_exist(out["image_path"], "HAM10000")

    return out.reset_index(drop=True)


def _assert_paths_exist(image_paths: pd.Series, dataset_name: str) -> None:
    """Helper that verifies every image in Series actually exists given their path."""
    missing = [p for p in map(Path, image_paths) if not p.exists()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:3])
        raise FileNotFoundError(
            f"{dataset_name}: {len(missing)} images missing. Examples: {preview}"
        )
