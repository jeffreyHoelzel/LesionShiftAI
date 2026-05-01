"""runtime.py

Logic to create run directories and write output JSON data.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from lesionshiftai.core.config import ExperimentConfig


def create_run_dir(
    cfg: ExperimentConfig,
    config_path: str | Path
) -> Path:
    """
    Creates a timestamped experiment run directory and copies the configuration file.

    Parameters
    ------------
        cfg : ExperimentConfig
            Experiment configuration containing the output root and experiment name.
        config_path : str | Path
            Path to the configuration file to copy into the run directory.

    Returns
    --------
        run_dir : Path
            Path to the created experiment run directory.

    Raises
    -------
        FileExistsError
            Raised when the timestamped run directory already exists.
        OSError
            Raised when run directories cannot be created or the configuration file cannot be copied.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / cfg.name / stamp
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.yml")
    return run_dir


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """
    Writes a dictionary payload to disk as formatted JSON.

    Parameters
    ------------
        path : str | Path
            Destination path for the JSON file.
        payload : Dict[str, Any]
            Dictionary payload to serialize and write.

    Returns
    --------
        None : None
            This function does not return a value.

    Raises
    -------
        TypeError
            Raised when the payload contains values that cannot be serialized to JSON.
        OSError
            Raised when the destination file cannot be written.
    """
    path = Path(path)
    path.write_text(json.dumps(
        payload, indent=2), encoding="utf-8"
    )
