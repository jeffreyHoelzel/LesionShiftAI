import sys
import types
import uuid
from pathlib import Path

import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_ROOT = ROOT / "src"
SCRIPTS_ROOT = ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


try:
    import cv2  # type: ignore
except ModuleNotFoundError:
    cv2 = types.SimpleNamespace()
    cv2.IMREAD_COLOR = 1
    cv2.COLOR_BGR2RGB = 4

    def _imread(path: str, _flags: int = 1):
        p = Path(path)
        if not p.exists():
            return None
        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        return rgb[:, :, ::-1]

    def _cvt_color(image: np.ndarray, code: int):
        if code == cv2.COLOR_BGR2RGB:
            return image[:, :, ::-1]
        return image

    cv2.imread = _imread
    cv2.cvtColor = _cvt_color
    sys.modules["cv2"] = cv2


try:
    import timm  # type: ignore
except ModuleNotFoundError:
    import torch.nn as nn

    class _FallbackBackbone(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2, 3), keepdim=True)

    timm = types.SimpleNamespace(
        create_model=lambda *args, **kwargs: _FallbackBackbone()
    )
    sys.modules["timm"] = timm


try:
    import torchvision  # type: ignore
except ModuleNotFoundError:
    import torch.nn as nn

    torchvision = types.ModuleType("torchvision")
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_models = types.ModuleType("torchvision.models")

    class _Compose:
        def __init__(self, ops):
            self.ops = ops

        def __call__(self, image):
            out = image
            for op in self.ops:
                out = op(out)
            return out

    class _Resize:
        def __init__(self, size):
            self.size = size

        def __call__(self, image):
            return image.resize((self.size[1], self.size[0]))

    class _RandomHorizontalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, image):
            return image

    class _RandomVerticalFlip:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, image):
            return image

    class _RandomApply:
        def __init__(self, transforms, p=0.5):
            self.transforms = transforms
            self.p = p

        def __call__(self, image):
            out = image
            for tfm in self.transforms:
                out = tfm(out)
            return out

    class _ColorJitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, image):
            return image

    class _ToTensor:
        def __call__(self, image):
            arr = np.asarray(image, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr)

    class _Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class _ResNet50Weights:
        IMAGENET1K_V2 = object()

    class _FallbackResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            return torch.zeros((batch, 1), dtype=torch.float32, device=x.device)

    def _resnet50(*, weights=None):
        return _FallbackResNet()

    tv_transforms.Compose = _Compose
    tv_transforms.Resize = _Resize
    tv_transforms.RandomHorizontalFlip = _RandomHorizontalFlip
    tv_transforms.RandomVerticalFlip = _RandomVerticalFlip
    tv_transforms.RandomApply = _RandomApply
    tv_transforms.ColorJitter = _ColorJitter
    tv_transforms.ToTensor = _ToTensor
    tv_transforms.Normalize = _Normalize

    tv_models.resnet50 = _resnet50
    tv_models.ResNet50_Weights = _ResNet50Weights

    torchvision.transforms = tv_transforms
    torchvision.models = tv_models
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = tv_transforms
    sys.modules["torchvision.models"] = tv_models


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    torch.manual_seed(123)
    np.random.seed(123)


@pytest.fixture
def tmp_path() -> Path:
    """Use a workspace-local temp dir to avoid host temp ACL issues on Windows."""
    root = ROOT / ".tmp_pytest_local"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"case_{uuid.uuid4().hex[:10]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


@pytest.fixture
def synthetic_dataset_factory(tmp_path: Path):
    def _factory(
        n_isic: int = 40,
        n_ham: int = 20,
        grouped: bool = True,
    ) -> tuple[Path, Path]:
        root = tmp_path / f"dataset_{uuid.uuid4().hex[:8]}"
        isic_root = root / "ISIC 2019"
        ham_root = root / "HAM10000"

        isic_image_dir = isic_root / "train images"
        ham_image_dir = ham_root / "images"
        isic_image_dir.mkdir(parents=True, exist_ok=True)
        ham_image_dir.mkdir(parents=True, exist_ok=True)

        isic_rows = []
        for idx in range(n_isic):
            sample_id = f"ISIC_{idx:05d}"
            label = idx % 2
            patient_id = f"P{idx // 2:03d}" if grouped else f"P{idx:03d}"
            image_path = isic_image_dir / f"{sample_id}.jpg"
            _save_rgb_image(image_path, color=30 + (idx % 200))
            isic_rows.append(
                {
                    "isic_id": sample_id,
                    "patient_id": patient_id,
                    "target": label,
                }
            )

        pd.DataFrame(isic_rows).to_csv(
            isic_root / "train-metadata.csv", index=False
        )

        ham_classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        ham_rows = []
        for idx in range(n_ham):
            sample_id = f"HAM_{idx:05d}"
            class_name = ham_classes[idx % len(ham_classes)]
            image_path = ham_image_dir / f"{sample_id}.jpg"
            _save_rgb_image(image_path, color=60 + (idx % 150))

            row = {name: 0 for name in ham_classes}
            row["image"] = sample_id
            row[class_name] = 1
            ham_rows.append(row)

        pd.DataFrame(ham_rows).to_csv(ham_root / "GroundTruth.csv", index=False)
        return isic_root, ham_root

    return _factory


@pytest.fixture
def synthetic_dataset_roots(synthetic_dataset_factory):
    return synthetic_dataset_factory()


@pytest.fixture
def write_config_factory(tmp_path: Path):
    def _factory(
        *,
        config_name: str = "exp.yml",
        experiment_name: str = "test_experiment",
        output_root: Path | None = None,
        isic_root: Path,
        ham_root: Path,
        epochs: int = 1,
        batch_size: int = 4,
        num_workers: int = 0,
        image_size: int = 64,
        val_size: float = 0.2,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        deterministic: bool = True,
        seed: int = 42,
    ) -> Path:
        out_root = output_root or (tmp_path / "outputs")
        payload = {
            "experiment_name": experiment_name,
            "output_root": str(out_root),
            "seed": int(seed),
            "deterministic": bool(deterministic),
            "data": {
                "isic_root": str(isic_root),
                "ham_root": str(ham_root),
                "image_size": int(image_size),
                "val_size": float(val_size),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "pin_memory": False,
            },
            "train": {
                "epochs": int(epochs),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "warmup_epochs": int(warmup_epochs),
                "min_lr": float(min_lr),
            },
        }
        cfg_path = tmp_path / config_name
        cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return cfg_path

    return _factory


@pytest.fixture
def tiny_binary_model_class():
    import torch.nn as nn

    class TinyBinaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat_mean = x.float().mean(dim=(1, 2, 3))
            return (flat_mean * self.scale) + self.bias

    return TinyBinaryModel


@pytest.fixture
def assert_has_metric_keys():
    def _assert(payload: dict) -> None:
        required = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        }
        missing = required.difference(payload.keys())
        assert not missing, f"Missing metric keys: {sorted(missing)}"

    return _assert


def _save_rgb_image(path: Path, color: int) -> None:
    arr = np.full((16, 16, 3), fill_value=int(color), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")
