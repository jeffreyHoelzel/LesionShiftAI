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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / cfg.name / stamp
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config.yml")
    return run_dir


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(
        payload, indent=2), encoding="utf-8"
    )
