import os
from dataclasses import dataclass
from typing import Any, List
import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistState:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_dist() -> DistState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # only one GPU running training, eval, inference, etc.
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistState(False, 0, 1, 0, device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # determine if GPUs available before distribution
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, init_method="env://")
    return DistState(True, rank, world_size, local_rank, device)


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(obj: Any) -> List[Any]:
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered
