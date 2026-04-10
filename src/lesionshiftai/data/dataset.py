from pathlib import Path
from typing import Any, Dict
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
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
