import os
from pathlib import Path

import pytest
import yaml

from lesionshiftai.core.config import (
    DataConfig,
    ExperimentConfig,
    TrainConfig,
    _expand_path,
    _val_config,
    load_config,
)


pytestmark = pytest.mark.unit


def test_expand_path_replaces_user_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER", "alice")
    expanded = _expand_path("/scratch/$USER/${USER}/<USER>/run")
    assert expanded.as_posix() == "/scratch/alice/alice/alice/run"


def test_load_config_populates_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yml"
    payload = {
        "experiment_name": "unit_cfg",
        "output_root": str(tmp_path / "out"),
        "data": {
            "isic_root": str(tmp_path / "isic"),
            "ham_root": str(tmp_path / "ham"),
        },
    }
    cfg_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config(cfg_path)

    assert cfg.name == "unit_cfg"
    assert cfg.output_root == tmp_path / "out"
    assert cfg.seed == 42
    assert cfg.deterministic is True
    assert cfg.data.batch_size == 32
    assert cfg.train.epochs == 20


@pytest.mark.parametrize(
    ("mutator", "message_fragment"),
    [
        (lambda c: setattr(c.data, "val_size", 0.0), "val_size"),
        (lambda c: setattr(c.data, "batch_size", 0), "batch_size"),
        (lambda c: setattr(c.data, "image_size", 32), "image_size"),
        (lambda c: setattr(c.train, "epochs", 0), "epochs"),
        (lambda c: setattr(c.train, "lr", 0.0), "lr"),
        (lambda c: setattr(c.train, "weight_decay", -0.1), "weight_decay"),
        (lambda c: setattr(c.train, "warmup_epochs", -1), "warmup_epochs"),
        (lambda c: setattr(c.train, "min_lr", -1.0), "min_lr"),
    ],
)
def test_val_config_rejects_invalid_ranges(mutator, message_fragment: str) -> None:
    cfg = ExperimentConfig(
        name="x",
        output_root=Path("out"),
        seed=42,
        deterministic=True,
        data=DataConfig(isic_root=Path("isic"), ham_root=Path("ham")),
        train=TrainConfig(),
    )
    mutator(cfg)
    with pytest.raises(ValueError) as exc:
        _val_config(cfg)
    assert message_fragment in str(exc.value)


def test_val_config_rejects_warmup_above_epochs() -> None:
    cfg = ExperimentConfig(
        name="x",
        output_root=Path("out"),
        seed=42,
        deterministic=True,
        data=DataConfig(isic_root=Path("isic"), ham_root=Path("ham")),
        train=TrainConfig(epochs=2, warmup_epochs=3),
    )
    with pytest.raises(ValueError) as exc:
        _val_config(cfg)
    assert "warmup_epochs" in str(exc.value)


def test_val_config_rejects_min_lr_above_lr() -> None:
    cfg = ExperimentConfig(
        name="x",
        output_root=Path("out"),
        seed=42,
        deterministic=True,
        data=DataConfig(isic_root=Path("isic"), ham_root=Path("ham")),
        train=TrainConfig(lr=1e-4, min_lr=1e-3),
    )
    with pytest.raises(ValueError) as exc:
        _val_config(cfg)
    assert "min_lr" in str(exc.value)
