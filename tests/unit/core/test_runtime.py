from datetime import datetime
from pathlib import Path

import pytest

from lesionshiftai.core.config import DataConfig, ExperimentConfig, TrainConfig
from lesionshiftai.core.runtime import create_run_dir, write_json


pytestmark = pytest.mark.unit


class _FixedNow:
    def strftime(self, _fmt: str) -> str:
        return "20260101_010203"


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _FixedNow()


def _cfg(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="runtime_test",
        output_root=tmp_path / "outputs",
        seed=42,
        deterministic=True,
        data=DataConfig(isic_root=tmp_path / "isic", ham_root=tmp_path / "ham"),
        train=TrainConfig(),
    )


def test_create_run_dir_makes_expected_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.core.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "datetime", _FixedDateTime)
    cfg = _cfg(tmp_path)
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("name: test\n", encoding="utf-8")

    run_dir = create_run_dir(cfg, config_path)

    assert run_dir.exists()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "metrics").is_dir()
    assert (run_dir / "predictions").is_dir()
    assert (run_dir / "config.yml").read_text(encoding="utf-8") == "name: test\n"


def test_create_run_dir_raises_on_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.core.runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "datetime", _FixedDateTime)
    cfg = _cfg(tmp_path)
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("name: test\n", encoding="utf-8")

    _ = create_run_dir(cfg, config_path)
    with pytest.raises(FileExistsError):
        create_run_dir(cfg, config_path)


def test_write_json_writes_pretty_json(tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    write_json(out, {"a": 1, "b": [1, 2]})
    text = out.read_text(encoding="utf-8")
    assert '"a": 1' in text
    assert text.strip().startswith("{")
