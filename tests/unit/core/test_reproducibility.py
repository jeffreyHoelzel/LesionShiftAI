import os
import random

import numpy as np
import pytest
import torch

from lesionshiftai.core.reproducibility import init_generator, seed_worker, set_seed


pytestmark = pytest.mark.unit


def test_set_seed_sets_global_deterministic_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"deterministic": None, "warn_only": None}

    def _fake_use_det(enabled: bool, warn_only: bool = False) -> None:
        calls["deterministic"] = enabled
        calls["warn_only"] = warn_only

    monkeypatch.setattr(torch, "use_deterministic_algorithms", _fake_use_det)

    set_seed(777, deterministic=True)

    assert os.environ["PYTHONHASHSEED"] == "777"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == {"deterministic": True, "warn_only": False}
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_worker_uses_torch_initial_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch, "initial_seed", lambda: 12345)
    seed_worker(0)

    a = np.random.randint(0, 1000)
    b = random.randint(0, 1000)

    seed_worker(0)
    assert np.random.randint(0, 1000) == a
    assert random.randint(0, 1000) == b


def test_init_generator_reproducible() -> None:
    gen_a = init_generator(88)
    gen_b = init_generator(88)
    a = torch.rand((4,), generator=gen_a)
    b = torch.rand((4,), generator=gen_b)
    assert torch.allclose(a, b)
