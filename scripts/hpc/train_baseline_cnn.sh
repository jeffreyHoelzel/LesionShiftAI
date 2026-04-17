#!/bin/bash
#SBATCH --job-name=train-baseline-cnn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h200:2
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/train_baseline_cnn/output/%x_%j.out
#SBATCH --error=/scratch/%u/train_baseline_cnn/error/%x_%j.err

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX=${ENV_PREFIX:-/scratch/$USER/conda/envs/lesionshiftai}
conda activate "$ENV_PREFIX"

NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE##*:}}
if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
    # let torchrun infer count from visible GPUs when SLURM format is non-numeric.
    NPROC_PER_NODE="gpu"
fi

srun torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    lesionshiftai.pyz train-baseline \
    --config baseline_cnn.yml
