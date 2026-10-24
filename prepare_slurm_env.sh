#!/usr/bin/env bash
# Source this file inside your allocation: source ./prepare_babel.sh
# Environment setup only; no model settings and no training launch.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source ./prepare_babel.sh' >&2
  exit 1
fi

CONDA_ROOT="${CONDA_ROOT:-/home/iapostol/miniconda3}"
CONDA_ENV="${CONDA_ENV:-sdft_babel}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh" || return 1
conda activate "$CONDA_ENV" || return 1

export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/iapostol}"
export SDFT_OUTPUT_ROOT="${SDFT_OUTPUT_ROOT:-${SCRATCH_ROOT}/sdft_runs}"
# Persistent caches avoid home quota use and repeated downloads after each job.
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
export HF_HUB_DISABLE_XET=1
export HF_XET_DISABLE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TRANSFORMERS_NO_TORCHVISION=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" || return 1

# Preserve Slurm's GPU visibility and explicit user overrides.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 50000))}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    BABEL_HOSTS="$(scontrol show hostnames "$SLURM_JOB_NODELIST")" || return 1
    export MASTER_ADDR="${BABEL_HOSTS%%$'\n'*}"
    unset BABEL_HOSTS
  else
    export MASTER_ADDR=127.0.0.1
  fi
fi

export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-2}"
export NCCL_MIN_CHANNELS="${NCCL_MIN_CHANNELS:-32}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
# Match your Babel eval commands; override either flag to re-enable transport.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

printf 'Babel environment ready. Output root: %s\n' "$SDFT_OUTPUT_ROOT"
