#!/bin/bash
#SBATCH --job-name=train-ensemble-cnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h200:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/train_ensemble_cnn_fold/output/%x_%j.out
#SBATCH --error=/scratch/%u/train_ensemble_cnn_fold/error/%x_%j.err

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX=${ENV_PREFIX:-/scratch/$USER/conda/envs/lesionshiftai}
conda activate "$ENV_PREFIX"

# point to correct dependencies in environment
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# required for deterministic CUDA linear algebra when cfg.deterministic=true
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

NUM_FOLDS=${ENSEMBLE_NUM_FOLDS:-5}
: "${ENSEMBLE_RUN_ID:?Set ENSEMBLE_RUN_ID to a shared ensemble run identifier}"

FOLD_ARGS=()
if [[ -n "${FOLD_INDEX:-}" ]]; then
    if (( FOLD_INDEX < 0 || FOLD_INDEX >= NUM_FOLDS )); then
        echo "Invalid fold index ${FOLD_INDEX} for ENSEMBLE_NUM_FOLDS=${NUM_FOLDS}"
        exit 1
    fi
    # optional one-fold override for targeted retries/debugging
    FOLD_ARGS=(--fold-index "$FOLD_INDEX")
fi

NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE##*:}}
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
    # let torchrun infer count from visible GPUs when SLURM format is non-numeric
    NPROC_PER_NODE="gpu"
fi

srun torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    lesionshiftai.pyz train-ensemble \
    --config baseline_cnn.yml \
    --num-folds "$NUM_FOLDS" \
    --ensemble-run-id "$ENSEMBLE_RUN_ID" \
    "${FOLD_ARGS[@]}"
