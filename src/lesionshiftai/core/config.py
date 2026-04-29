"""config.py

Core configuration of types used throughout model 
training, validation, and testing.
"""
import getpass
import os
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(slots=True)
class DataConfig:
    """
    Stores dataset and DataLoader configuration values.

    Parameters
    ------------
        isic_root : Path
            Root directory for the ISIC dataset.
        ham_root : Path
            Root directory for the HAM dataset.
        image_size : int
            Target image size used during preprocessing.
        val_size : float
            Fraction of data reserved for validation.
        batch_size : int
            Number of samples loaded per batch.
        num_workers : int
            Number of worker processes used by each DataLoader.
        pin_memory : bool
            Whether DataLoader should pin memory for faster GPU transfer.

    Returns
    --------
        DataConfig : DataConfig
            Dataclass instance containing data configuration values.

    Raises
    -------
        TypeError
            Raised when required fields are missing or incompatible values are provided.
    """
    isic_root: Path
    ham_root: Path
    image_size: int = 224
    val_size: float = 0.20
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass(slots=True)
class TrainConfig:
    """
    Stores training hyperparameter configuration values.

    Parameters
    ------------
        epochs : int
            Number of training epochs.
        lr : float
            Initial learning rate.
        weight_decay : float
            Weight decay used by the optimizer.
        warmup_epochs : int
            Number of warmup epochs before the main learning rate schedule.
        min_lr : float
            Minimum learning rate used during scheduling.

    Returns
    --------
        TrainConfig : TrainConfig
            Dataclass instance containing training configuration values.

    Raises
    -------
        TypeError
            Raised when incompatible values are provided.
    """
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    min_lr: float = 1e-6


@dataclass(slots=True)
class ExperimentConfig:
    """
    Stores the full experiment configuration.

    Parameters
    ------------
        name : str
            Name of the experiment.
        output_root : Path
            Root directory where experiment outputs are saved.
        seed : int
            Random seed used for reproducibility.
        deterministic : bool
            Whether deterministic training behavior should be enabled.
        data : DataConfig
            Dataset and DataLoader configuration.
        train : TrainConfig
            Training hyperparameter configuration.

    Returns
    --------
        ExperimentConfig : ExperimentConfig
            Dataclass instance containing the full experiment configuration.

    Raises
    -------
        TypeError
            Raised when required fields are missing or incompatible values are provided.
    """
    name: str
    output_root: Path
    seed: int
    deterministic: bool
    data: DataConfig
    train: TrainConfig


def _expand_path(raw_path: str | Path) -> Path:
    """Expands user and environment variables in a filesystem path."""
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
    """
    Loads and validates an experiment configuration from a YAML file.

    Parameters
    ------------
        path : str | Path
            Path to the YAML configuration file.

    Returns
    --------
        cfg : ExperimentConfig
            Validated experiment configuration object.

    Raises
    -------
        KeyError
            Raised when required configuration keys are missing.
        ValueError
            Raised when configuration values fail validation.
        OSError
            Raised when the configuration file cannot be read.
        yaml.YAMLError
            Raised when the YAML file cannot be parsed.
    """
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
            weight_decay=float(train_raw.get("weight_decay", 1e-4)),
            warmup_epochs=int(train_raw.get("warmup_epochs", 3)),
            min_lr=float(train_raw.get("min_lr", 1e-6))
        )
    )
    _val_config(cfg)
    return cfg


def _val_config(cfg: ExperimentConfig) -> None:
    """Validates an ExperimentConfig object."""
    if not 0.0 < cfg.data.val_size < 0.5:
        raise ValueError("data.val_size must be between 0 and 0.5")
    if cfg.data.batch_size < 1:
        raise ValueError("data.batch_size must be >= 1")
    if cfg.data.image_size < 64:
        raise ValueError("data.image_size must be >= 64")
    if cfg.train.epochs < 1:
        raise ValueError("train.epochs must be >= 1")
    if cfg.train.lr <= 0.0:
        raise ValueError("train.lr must be > 0")
    if cfg.train.weight_decay < 0.0:
        raise ValueError("train.weight_decay must be >= 0")
    if cfg.train.warmup_epochs < 0:
        raise ValueError("train.warmup_epochs must be >= 0")
    if cfg.train.warmup_epochs > cfg.train.epochs:
        raise ValueError("train.warmup_epochs must be <= train.epochs")
    if cfg.train.min_lr < 0.0:
        raise ValueError("train.min_lr must be >= 0")
    if cfg.train.min_lr > cfg.train.lr:
        raise ValueError("train.min_lr must be <= train.lr")
