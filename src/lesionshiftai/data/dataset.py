"""dataset.py

Main Dataset class used to handle ISIC 2019 and HAM10000 224x224 images.
"""
from pathlib import Path
from typing import Any, Dict
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    """
    Loads skin lesion images and labels from a metadata DataFrame.

    Parameters
    ------------
        df : pd.DataFrame
            Metadata DataFrame containing image paths, labels, sample IDs, and dataset names.
        transform : Any
            Optional image transform applied after loading and RGB conversion.

    Returns
    --------
        LesionDataset : LesionDataset
            Dataset instance for loading transformed lesion samples.

    Raises
    -------
        TypeError
            Raised when required fields are missing or incompatible values are provided.
    """

    def __init__(self, df: pd.DataFrame, transform: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Loads and returns one transformed lesion sample."""
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return {
            "image": image,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "sample_id": row["sample_id"],
            "dataset": row["dataset"]
        }
