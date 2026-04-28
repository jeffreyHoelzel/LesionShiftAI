from pathlib import Path

import pandas as pd
import pytest

from lesionshiftai.data.labels import HAM_CLASS_COLUMNS
from lesionshiftai.data.metadata import load_ham_metadata, load_isic_metadata


pytestmark = pytest.mark.unit


def test_load_isic_metadata_success(synthetic_dataset_roots: tuple[Path, Path]) -> None:
    isic_root, _ = synthetic_dataset_roots
    df = load_isic_metadata(isic_root)

    assert {"sample_id", "patient_id", "image_path", "label", "source_class", "dataset"}.issubset(df.columns)
    assert set(df["dataset"].unique()) == {"isic2019"}
    assert set(df["label"].unique()) == {0, 1}


def test_load_isic_metadata_missing_columns(tmp_path: Path) -> None:
    isic_root = tmp_path / "isic"
    (isic_root / "train images").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"isic_id": "x"}]).to_csv(isic_root / "train-metadata.csv", index=False)

    with pytest.raises(ValueError) as exc:
        load_isic_metadata(isic_root, strict_images=False)
    assert "missing columns" in str(exc.value)


def test_load_isic_metadata_rejects_non_binary_labels(tmp_path: Path) -> None:
    isic_root = tmp_path / "isic"
    img_dir = isic_root / "train images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "A.jpg").write_bytes(b"x")

    pd.DataFrame(
        [{"isic_id": "A", "patient_id": "P1", "target": 2}]
    ).to_csv(isic_root / "train-metadata.csv", index=False)

    with pytest.raises(ValueError) as exc:
        load_isic_metadata(isic_root, strict_images=False)
    assert "non-binary" in str(exc.value)


def test_load_isic_metadata_strict_image_check(tmp_path: Path) -> None:
    isic_root = tmp_path / "isic"
    (isic_root / "train images").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"isic_id": "A", "patient_id": "P1", "target": 0},
            {"isic_id": "B", "patient_id": "P2", "target": 1},
        ]
    ).to_csv(isic_root / "train-metadata.csv", index=False)

    with pytest.raises(FileNotFoundError):
        load_isic_metadata(isic_root, strict_images=True)


def test_load_ham_metadata_success(synthetic_dataset_roots: tuple[Path, Path]) -> None:
    _, ham_root = synthetic_dataset_roots
    df = load_ham_metadata(ham_root)

    assert {"sample_id", "patient_id", "image_path", "label", "source_class", "dataset"}.issubset(df.columns)
    assert set(df["dataset"].unique()) == {"ham10000"}
    assert set(df["source_class"]).issubset(set(HAM_CLASS_COLUMNS))


def test_load_ham_metadata_missing_columns(tmp_path: Path) -> None:
    ham_root = tmp_path / "ham"
    (ham_root / "images").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"image": "x"}]).to_csv(ham_root / "GroundTruth.csv", index=False)

    with pytest.raises(ValueError) as exc:
        load_ham_metadata(ham_root, strict_images=False)
    assert "missing columns" in str(exc.value)


def test_load_ham_metadata_rejects_bad_one_hot(tmp_path: Path) -> None:
    ham_root = tmp_path / "ham"
    (ham_root / "images").mkdir(parents=True, exist_ok=True)
    row = {name: 0 for name in HAM_CLASS_COLUMNS}
    row["image"] = "H1"
    row["MEL"] = 1
    row["NV"] = 1
    pd.DataFrame([row]).to_csv(ham_root / "GroundTruth.csv", index=False)

    with pytest.raises(ValueError) as exc:
        load_ham_metadata(ham_root, strict_images=False)
    assert "invalid one-hot" in str(exc.value)


def test_load_ham_metadata_strict_images(tmp_path: Path) -> None:
    ham_root = tmp_path / "ham"
    (ham_root / "images").mkdir(parents=True, exist_ok=True)
    row = {name: 0 for name in HAM_CLASS_COLUMNS}
    row["image"] = "H1"
    row["MEL"] = 1
    pd.DataFrame([row]).to_csv(ham_root / "GroundTruth.csv", index=False)

    with pytest.raises(FileNotFoundError):
        load_ham_metadata(ham_root, strict_images=True)
