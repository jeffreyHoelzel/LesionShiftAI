import numpy as np
import pytest
import torch

from lesionshiftai.data.transforms import build_eval_transform, build_train_transform


pytestmark = pytest.mark.unit


def test_build_train_transform_output_shape() -> None:
    tfm = build_train_transform(32)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    out = tfm(image=image)
    assert isinstance(out["image"], torch.Tensor)
    assert tuple(out["image"].shape) == (3, 32, 32)


def test_build_eval_transform_output_shape() -> None:
    tfm = build_eval_transform(28)
    image = np.zeros((12, 8, 3), dtype=np.uint8)
    out = tfm(image=image)
    assert isinstance(out["image"], torch.Tensor)
    assert tuple(out["image"].shape) == (3, 28, 28)
