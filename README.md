# LesionShiftAI

LesionShiftAI is a research benchmark for cross-dataset skin lesion classification under dataset shift. The project trains models on ISIC 2019 and evaluates external generalization on HAM10000.

The repository currently supports three training/evaluation pipelines:

- Baseline CNN
- Ensemble of CNNs (k-fold members + aggregate predictions)
- Vision Transformer (ViT-B16)

Each training script performs:

1. ISIC training
2. ISIC validation
3. HAM10000 external test
4. Artifact export (checkpoints, predictions, metrics, ROC/PR curves, generalization gap)

## Repository Layout

```text
LesionShiftAI/
|- config/                 # baseline_cnn.yml and vit_b16.yml
|- scripts/
|  |- hpc/                 # SLURM launch scripts for Monsoon
|  |- train_baseline_cnn.py
|  |- train_ensemble_member_cnn.py
|  |- train_vit.py
|- src/lesionshiftai/      # core, data, models, train, eval modules
|- tests/                  # unit and integration tests
|- .github/workflows/ci.yml
|- pyproject.toml
|- environment.yml
`- dist/lesionshiftai.pyz  # zipapp launcher used on HPC
```

## Dataset Expectations

LesionShiftAI expects these files/directories:

```text
ISIC 2019/
|- train-metadata.csv
`- train images/
   `- <isic_id>.jpg

HAM10000/
|- GroundTruth.csv
`- images/
   `- <image>.jpg
```

## Environment Setup

### Local Development with uv

Use Python 3.12 locally.

```bash
uv sync --extra dev --python 3.12
```

Run commands without activating the virtual environment:

```bash
uv run python --version
uv run pytest -m unit --no-cov
```

### Monsoon HPC with Conda

Create a shared environment in scratch (recommended path matches SLURM scripts):

```bash
module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml -p /scratch/$USER/conda/envs/lesionshiftai
conda activate /scratch/$USER/conda/envs/lesionshiftai
```

If the environment already exists:

```bash
conda env update -f environment.yml -p /scratch/$USER/conda/envs/lesionshiftai --prune
```

## HPC Runtime Preparation (Monsoon)

The SLURM scripts in `scripts/hpc` invoke `lesionshiftai.pyz` and config files by filename only, so stage them in your working directory before submitting jobs:

```bash
uv run python scripts/build_pyz.py
scp dist/lesionshiftai.pyz <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp config/baseline_cnn.yml <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp config/vit_b16.yml <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp scripts/hpc/train_baseline_cnn.sh <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp scripts/hpc/train_ensemble_cnn.sh <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp scripts/hpc/train_vit.sh <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
ssh <USER>@monsoon.hpc.nau.edu # make sure you are connected to NAU VPN if not on campus
cd lesionshiftai
dos2unix *.sh # convert to Linux format if coming from Windows machine
chmod +x *.sh
```

## Configure Experiments

Edit the staged `baseline_cnn.yml` and `vit_b16.yml` before launching jobs.

Required fields to set:

- `experiment_name`: unique name for each run family
- `output_root`: recommended `/scratch/$USER/lesionshiftai/outputs`
- `data.isic_root`: absolute path to ISIC 2019 root
- `data.ham_root`: absolute path to HAM10000 root

Common tuning fields:

- `data.batch_size`, `data.num_workers`, `data.image_size`
- `train.epochs`, `train.lr`, `train.weight_decay`
- ViT only: `train.warmup_epochs`, `train.min_lr`

## Train and Evaluate on HPC

All commands below should be run from the directory that contains:

- `lesionshiftai.pyz`
- `baseline_cnn.yml`
- `vit_b16.yml`
- `scripts/hpc/`

### Baseline CNN

```bash
sbatch train_baseline_cnn.sh
```

Primary artifacts:

- `checkpoints/best.pt`
- `predictions/val_final.csv`
- `predictions/ham_test.csv`
- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `metrics/generalization_gap.json`

### Ensemble of CNNs

`ENSEMBLE_RUN_ID` is required and must be shared across fold jobs.

Run all folds in one job:

```bash
export ENSEMBLE_RUN_ID=ens_$(date +%Y%m%d_%H%M%S)
sbatch train_ensemble_cnn.sh
```

Run a single fold (for retry/debug):

```bash
export ENSEMBLE_RUN_ID=ens_20260429_rerun
FOLD_INDEX=0 sbatch train_ensemble_cnn.sh
```

Optional fold count override:

```bash
ENSEMBLE_NUM_FOLDS=5 sbatch train_ensemble_cnn.sh
```

Primary artifacts:

- `members/fold_<k>/...` for each member
- `ensemble/predictions/isic_val_aggregate_predictions.csv`
- `ensemble/predictions/ham_test_aggregate_predictions.csv`
- `ensemble/metrics/isic_val_aggregate_metrics.json`
- `ensemble/metrics/ham_test_aggregate_metrics.json`
- `ensemble/metrics/generalization_gap.json`

### Vision Transformer (ViT-B16)

```bash
sbatch train_vit.sh
```

Primary artifacts:

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `predictions/val_final.csv`
- `predictions/ham_test.csv`
- `metrics/val_metrics.json`
- `metrics/test_metrics.json`
- `metrics/generalization_gap.json`

## Optional Local Training Commands

For local smoke runs (single process):

```bash
uv run python scripts/train_baseline_cnn.py --config config/baseline_cnn.yml --threshold 0.5
uv run python scripts/train_ensemble_member_cnn.py --config config/baseline_cnn.yml --num-folds 5 --ensemble-run-id local_smoke --threshold 0.5
uv run python scripts/train_vit.py --config config/vit_b16.yml --threshold 0.5
```

## Testing and Code Correctness

### Local Test Commands

Fast feedback:

```bash
uv run pytest -m unit --no-cov
```

Pipeline smoke tests:

```bash
uv run pytest -m integration --no-cov
```

Full validation with coverage gate:

```bash
uv run pytest
```

The project enforces `--cov-fail-under=85` through `pyproject.toml`.

### CI Pipeline

GitHub Actions workflow: `.github/workflows/ci.yml`

The CI job runs on:

- Pull requests targeting `main`
- Manual trigger (`workflow_dispatch`)

CI steps:

1. Install package + test dependencies
2. Run unit tests (`pytest -m unit --no-cov`)
3. Run integration tests (`pytest -m integration --no-cov`)
4. Run full suite with coverage gate (`pytest`)

## Author

Jeffrey Hoelzel Jr.

## Disclaimer

**LesionShiftAI is a research benchmarking project and is not a clinical diagnostic tool.**
