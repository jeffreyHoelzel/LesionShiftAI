from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "script_path",
    [
        Path("scripts/hpc/train_baseline_cnn.sh"),
        Path("scripts/hpc/train_ensemble_cnn.sh"),
        Path("scripts/hpc/train_vit.sh"),
    ],
)
def test_hpc_scripts_have_required_structure(script_path: Path) -> None:
    text = script_path.read_text(encoding="utf-8")

    assert "#!/bin/bash" in text
    assert "#SBATCH --partition=gpu" in text
    assert "srun torchrun" in text
    assert "lesionshiftai.pyz" in text


def test_ensemble_hpc_script_requires_run_id() -> None:
    text = Path("scripts/hpc/train_ensemble_cnn.sh").read_text(encoding="utf-8")
    assert "ENSEMBLE_RUN_ID" in text
    assert "--ensemble-run-id" in text
    assert "FOLD_INDEX" in text
