import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        # required for deterministic GEMM ops on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)


def seed_worker(_: int) -> None:
    """Seed worker wrapper to fit function signature expected in DataLoader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
