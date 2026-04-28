import importlib
import tempfile
import contextlib
import shutil
import zipfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_build_pyz_creates_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module("build_pyz")

    fake_root = tmp_path / "fake_project"
    (fake_root / "src" / "lesionshiftai").mkdir(parents=True, exist_ok=True)
    (fake_root / "scripts").mkdir(parents=True, exist_ok=True)
    (fake_root / "run").mkdir(parents=True, exist_ok=True)

    (fake_root / "src" / "lesionshiftai" / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "src" / "lesionshiftai" / "module.py").write_text("x=1\n", encoding="utf-8")
    (fake_root / "src" / "lesionshiftai" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (fake_root / "src" / "lesionshiftai" / "__pycache__" / "junk.pyc").write_bytes(b"x")

    for script_name in [
        "train_ensemble_member_cnn.py",
        "train_baseline_cnn.py",
        "train_vit.py",
        "smoke_data_pipeline.py",
    ]:
        (fake_root / "scripts" / script_name).write_text("def main():\n    pass\n", encoding="utf-8")

    (fake_root / "run" / "__main__.py").write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setattr(mod, "ROOT", fake_root)
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    tempfile.tempdir = None

    @contextlib.contextmanager
    def _fake_temporary_directory():
        td = tmp_path / "tmp_manual"
        td.mkdir(parents=True, exist_ok=True)
        try:
            yield str(td)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    monkeypatch.setattr(mod.tempfile, "TemporaryDirectory", _fake_temporary_directory)

    out = tmp_path / "dist" / "lesionshiftai.pyz"
    mod.build_pyz(out)

    assert out.exists()
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())

    assert "__main__.py" in names
    assert "lesionshiftai/module.py" in names
    assert all("__pycache__" not in name for name in names)
