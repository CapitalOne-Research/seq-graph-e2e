#!/bin/bash
# Camera-ready A6: Gowalla no-relation ablation, one single-GPU job per seed.
# Uses the reconstructed main-table flags (MAIN_TABLE_FLAGS=1; the reference checkpoints were lost).
#   cd brightkite && mkdir -p norel_logs && sbatch run_gowalla_norel.sbatch
# Resubmitting resumes unfinished seeds and skips finished ones.
#SBATCH --job-name=gowalla-norel
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=norel_logs/slurm-%A_%a.out

set -euo pipefail
# Supply --account and --partition to sbatch. An already activated environment
# is inherited; optionally initialize a different environment inside this job.
if [[ -n "${CONDA_SH:-}" || -n "${CONDA_ENV:-}" ]]; then
    : "${CONDA_SH:?Set CONDA_SH to your conda.sh path}"
    : "${CONDA_ENV:?Set CONDA_ENV to your environment name or path}"
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi
cd "${SLURM_SUBMIT_DIR}"
s=${SLURM_ARRAY_TASK_ID}
python joint.py --dataset gowalla --gnn_layer_types GCN,GCN --num_neighbors 10 --gnn_heads 2 \
    --enable_early_stopping --eval_batch_size 64 --batch_size 32 --num_negative_samples 100 \
    --no_relational --seed "${s}" --experiment_name "gowalla-norel_s${s}" --skip_if_exists --resume \
    > "norel_logs/gowalla_norel_seed${s}.log" 2>&1
