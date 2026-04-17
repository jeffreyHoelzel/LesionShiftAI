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
module load mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_PREFIX=${ENV_PREFIX:-/scratch/$USER/conda/envs/lesionshiftai}
mamba activate "$ENV_PREFIX"

# get number of GPUs being used on node
GPUS=${SLURM_GPUS_ON_NODE##*:}

srun torchrun \
    --standalone \
    --nproc_per_node="$GPUS" \
    lesionshiftai.pyz train-baseline \
    --config baseline_cnn.yml
