import argparse
from types import SimpleNamespace

import pytest

import run.__main__ as launcher


pytestmark = pytest.mark.unit


def test_resolve_command(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(main=lambda: "ok")

    def _fake_import(_name: str):
        return fake_module

    monkeypatch.setattr(launcher.importlib, "import_module", _fake_import)
    fn = launcher._resolve_command("train-baseline")
    assert fn() == "ok"


def test_main_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"invoked": False, "argv": None}

    def _fake_parse_args(self):
        return argparse.Namespace(command="train-baseline", args=["--config", "x.yml"])

    def _fake_resolve(_command: str):
        def _run():
            called["invoked"] = True
            called["argv"] = list(launcher.sys.argv)

        return _run

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _fake_parse_args)
    monkeypatch.setattr(launcher, "_resolve_command", _fake_resolve)

    launcher.main()

    assert called["invoked"] is True
    assert called["argv"] == ["train-baseline", "--config", "x.yml"]
