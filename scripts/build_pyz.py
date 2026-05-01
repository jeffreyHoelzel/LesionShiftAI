"""build_pyz.py

Runs the process of building the `dist/lesionshiftai.pyz` script for HPC usage.
"""
import argparse
import shutil
import tempfile
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _copytree(src: Path, dst: Path) -> None:
    """Copies file tree from source to destination."""
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
    )


def build_pyz(output: Path) -> None:
    """
    Runs the .pyz logic and orders script entry-points for easy HPC usage by CLI.

    Parameters
    ----------
        output : Path
            Where to write the `lesionshiftai.pyz` script to. Default is `dist/`.
    """
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "stage"
        stage.mkdir(parents=True, exist_ok=True)

        # app package
        _copytree(ROOT / "src" / "lesionshiftai", stage / "lesionshiftai")

        # script entry modules called by launcher
        shutil.copy2(ROOT / "scripts" / "train_ensemble_member_cnn.py",
                     stage / "train_ensemble_member_cnn.py")
        shutil.copy2(ROOT / "scripts" / "train_baseline_cnn.py",
                     stage / "train_baseline_cnn.py")
        shutil.copy2(ROOT / "scripts" / "train_vit.py",
                     stage / "train_vit.py")

        # zipapp entrypoint
        shutil.copy2(ROOT / "run" / "__main__.py", stage / "__main__.py")

        zipapp.create_archive(
            stage,
            target=output,
            interpreter="/usr/bin/env python3",
            compressed=True
        )

    print(f"Built: {output}")


def main() -> None:
    """Main function to run .pyz builder."""
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="dist/lesionshiftai.pyz", type=str)
    args = p.parse_args()
    build_pyz(Path(args.output))


if __name__ == "__main__":
    main()
