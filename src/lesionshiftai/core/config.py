import getpass
import os
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(slots=True)
class DataConfig:
    isic_root: Path
    ham_root: Path
    image_size: int = 224
    val_size: float = 0.20
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_root: Path
    seed: int
    deterministic: bool
    data: DataConfig
    train: TrainConfig


def _expand_path(raw_path: str | Path) -> Path:
    raw = str(raw_path)
    user = (
        os.environ.get("USER")
        or os.environ.get("USERNAME")
        or getpass.getuser()
    )
    expanded = (
        raw.replace("<USER>", user)
        .replace("${USER}", user)
        .replace("$USER", user)
    )
    expanded = os.path.expandvars(expanded)
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def load_config(path: str | Path) -> ExperimentConfig:
    """Experiment configuration API for loading and validating YML file."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    data_raw = raw.get("data", {})
    train_raw = raw.get("train", {})

    cfg = ExperimentConfig(
        name=str(raw.get("experiment_name", "sample_experiment")),
        output_root=_expand_path(raw.get("output_root", "outputs")),
        seed=int(raw.get("seed", 42)),
        deterministic=bool(raw.get("deterministic", True)),
        data=DataConfig(
            isic_root=_expand_path(data_raw["isic_root"]),
            ham_root=_expand_path(data_raw["ham_root"]),
            image_size=int(data_raw.get("image_size", 224)),
            val_size=float(data_raw.get("val_size", 0.20)),
            batch_size=int(data_raw.get("batch_size", 32)),
            num_workers=int(data_raw.get("num_workers", 4)),
            pin_memory=bool(data_raw.get("pin_memory", True))
        ),
        train=TrainConfig(
            epochs=int(train_raw.get("epochs", 20)),
            lr=float(train_raw.get("lr", 3e-4)),
            weight_decay=float(train_raw.get("weight_decay", 1e-4))
        )
    )
    _val_config(cfg)
    return cfg


def _val_config(cfg: ExperimentConfig) -> None:
    """Helper to validate ExperimentConfig object was populated correctly."""
    if not 0.0 < cfg.data.val_size < 0.5:
        raise ValueError("data.val_size must be between 0 and 0.5")
    if cfg.data.batch_size < 1:
        raise ValueError("data.batch_size must be >= 1")
    if cfg.data.image_size < 64:
        raise ValueError("data.image_size must be >= 64")
