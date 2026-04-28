from typing import Any

import pytest
import torch

from lesionshiftai.core.distributed import DistState, all_gather_object, barrier, cleanup_dist, setup_dist


pytestmark = pytest.mark.unit


def test_dist_state_is_main_property() -> None:
    assert DistState(False, 0, 1, 0, torch.device("cpu")).is_main is True
    assert DistState(True, 1, 2, 1, torch.device("cpu")).is_main is False


def test_setup_dist_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    state = setup_dist()

    assert state.enabled is False
    assert state.rank == 0
    assert state.world_size == 1
    assert state.local_rank == 0
    assert state.device.type == "cpu"


def test_setup_dist_multi_process_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, Any] = {}

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    import lesionshiftai.core.distributed as dist_mod

    def _fake_init_process_group(*, backend: str, init_method: str) -> None:
        called["backend"] = backend
        called["init_method"] = init_method

    monkeypatch.setattr(dist_mod.dist, "init_process_group", _fake_init_process_group)

    state = setup_dist()

    assert state.enabled is True
    assert state.rank == 1
    assert state.world_size == 2
    assert state.device.type == "cpu"
    assert called == {"backend": "gloo", "init_method": "env://"}


def test_setup_dist_raises_on_bad_local_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "9")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError) as exc:
        setup_dist()
    assert "LOCAL_RANK" in str(exc.value)


def test_cleanup_dist_only_when_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.core.distributed as dist_mod

    calls = {"destroy": 0}
    monkeypatch.setattr(dist_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist_mod.dist,
        "destroy_process_group",
        lambda: calls.__setitem__("destroy", calls["destroy"] + 1),
    )

    cleanup_dist()
    assert calls["destroy"] == 1


def test_barrier_only_when_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.core.distributed as dist_mod

    calls = {"barrier": 0}
    monkeypatch.setattr(dist_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist_mod.dist,
        "barrier",
        lambda: calls.__setitem__("barrier", calls["barrier"] + 1),
    )

    barrier()
    assert calls["barrier"] == 1


def test_all_gather_object_without_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.core.distributed as dist_mod

    monkeypatch.setattr(dist_mod.dist, "is_available", lambda: False)
    payload = {"x": 1}
    assert all_gather_object(payload) == [payload]


def test_all_gather_object_with_dist(monkeypatch: pytest.MonkeyPatch) -> None:
    import lesionshiftai.core.distributed as dist_mod

    monkeypatch.setattr(dist_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(dist_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist_mod.dist, "get_world_size", lambda: 2)

    def _fake_all_gather_object(gathered, obj):
        gathered[0] = obj
        gathered[1] = {"replica": 1}

    monkeypatch.setattr(dist_mod.dist, "all_gather_object", _fake_all_gather_object)

    result = all_gather_object({"replica": 0})
    assert result == [{"replica": 0}, {"replica": 1}]
