"""distributed.py

Core DDP helper that maintains multi-GPU distributed training, 
validation, and testing.
"""
import os
from dataclasses import dataclass
from typing import Any, List
import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistState:
    """
    Stores distributed training process state.

    Parameters
    ------------
        enabled : bool
            Whether distributed training is enabled.
        rank : int
            Global process rank.
        world_size : int
            Total number of distributed processes.
        local_rank : int
            Process rank on the current node.
        device : torch.device
            Torch device assigned to the current process.

    Returns
    --------
        DistState : DistState
            Dataclass instance containing distributed process state.

    Raises
    -------
        TypeError
            Raised when required fields are missing or incompatible values are provided.
    """
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        """
        Checks whether the current process is the main process.

        Returns
        --------
            is_main : bool
                True when the current process rank is 0, otherwise False.
        """
        return self.rank == 0


def setup_dist() -> DistState:
    """
    Initializes distributed training state and selects the process device.

    Parameters
    ------------
        None : None
            This function does not accept parameters.

    Returns
    --------
        dist_state : DistState
            Distributed process state containing rank, world size, local rank, and device.

    Raises
    -------
        KeyError
            Raised when required distributed environment variables are missing.
        RuntimeError
            Raised when LOCAL_RANK is outside the range of visible CUDA devices.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # only one GPU running training, eval, inference, etc.
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistState(False, 0, 1, 0, device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # determine if GPUs available before distribution
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= device_count:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is out of range for {device_count} visible "
                "CUDA device(s). "
                f"(RANK={rank}, WORLD_SIZE={world_size}, "
                f"CUDA_VISIBLE_DEVICES={visible_devices}). "
                "Match torchrun --nproc_per_node to visible GPUs per task."
            )
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, init_method="env://")
    return DistState(True, rank, world_size, local_rank, device)


def cleanup_dist() -> None:
    """Destroys the active distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronizes all initialized distributed processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_gather_object(obj: Any) -> List[Any]:
    """
    Gathers a Python object from every distributed process.

    Parameters
    ------------
        obj : Any
            Python object to gather from the current process.

    Returns
    --------
        gathered : List[Any]
            List containing one gathered object from each distributed process.

    Raises
    -------
        RuntimeError
            Raised when distributed object gathering fails.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered
