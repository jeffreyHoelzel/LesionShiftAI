"""reproducibility.py

Handles all reproducibility aspects of training and testing 
like setting seeds and starting generators.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Sets random seeds across Python, NumPy, and PyTorch for reproducible execution.

    Parameters
    ------------
        seed : int
            Seed value used to initialize random number generators.
        deterministic : bool
            Whether to enable deterministic PyTorch and cuDNN behavior.

    Returns
    --------
        None : None
            This function does not return a value.

    Raises
    -------
        RuntimeError
            Raised when deterministic PyTorch algorithms are enabled but an operation does not support deterministic execution.
    """
    if deterministic:
        # required for deterministic
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
    """
    Seeds a DataLoader worker process.

    Parameters
    ------------
        _ : int
            Worker ID provided by the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def init_generator(seed: int) -> torch.Generator:
    """
    Initializes a seeded PyTorch random number generator.

    Parameters
    ------------
        seed : int
            Seed value used to initialize the generator.

    Returns
    --------
        gen : torch.Generator
            Seeded PyTorch generator.

    Raises
    -------
        RuntimeError
            Raised when PyTorch fails to initialize or seed the generator.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
