#!/usr/bin/env bash
#SBATCH --job-name=phi_article
#SBATCH --account=proj_1819
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b|type_c
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --array=0-48
#SBATCH --output=logs/campaign-%A_%a.out
#SBATCH --error=logs/campaign-%A_%a.err
#SBATCH --signal=B:TERM@120

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 NUM_WORKERS=0
export DATA_DIR="${DATA_DIR:-$PWD/assets/datasets}"
export MODEL_DIR="${MODEL_DIR:-$PWD/assets/models/Phi-4-mini-instruct}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
exec "${PHI_PYTHON:-.venv/bin/python}" -u hpc/run_job.py "${SLURM_ARRAY_TASK_ID:?}" \
    --manifest campaign_20260927.json
