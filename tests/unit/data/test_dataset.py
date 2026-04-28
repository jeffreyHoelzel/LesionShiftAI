from pathlib import Path

import pandas as pd
import pytest
import torch

from lesionshiftai.data.dataset import LesionDataset


pytestmark = pytest.mark.unit


def test_dataset_getitem_returns_expected_fields(synthetic_dataset_roots: tuple[Path, Path]) -> None:
    isic_root, _ = synthetic_dataset_roots
    df = pd.DataFrame(
        [
            {
                "sample_id": "ISIC_00000",
                "patient_id": "P000",
                "image_path": str(isic_root / "train images" / "ISIC_00000.jpg"),
                "label": 1,
                "source_class": "malignant",
                "dataset": "isic2019",
            }
        ]
    )

    ds = LesionDataset(df=df, transform=None)
    sample = ds[0]

    assert set(sample.keys()) == {"image", "label", "sample_id", "dataset"}
    assert isinstance(sample["label"], torch.Tensor)
    assert sample["label"].dtype == torch.float32
    assert sample["sample_id"] == "ISIC_00000"


def test_dataset_raises_when_image_missing(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "MISSING",
                "patient_id": "P0",
                "image_path": str(tmp_path / "missing.jpg"),
                "label": 0,
                "source_class": "benign",
                "dataset": "isic2019",
            }
        ]
    )
    ds = LesionDataset(df=df, transform=None)

    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_dataset_applies_transform(synthetic_dataset_roots: tuple[Path, Path]) -> None:
    isic_root, _ = synthetic_dataset_roots

    class _Transform:
        def __call__(self, *, image):
            return {"image": torch.tensor(image).float()}

    df = pd.DataFrame(
        [
            {
                "sample_id": "ISIC_00001",
                "patient_id": "P000",
                "image_path": str(isic_root / "train images" / "ISIC_00001.jpg"),
                "label": 0,
                "source_class": "benign",
                "dataset": "isic2019",
            }
        ]
    )
    ds = LesionDataset(df=df, transform=_Transform())
    sample = ds[0]
    assert isinstance(sample["image"], torch.Tensor)
