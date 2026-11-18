#!/usr/bin/env bash
# Reference Linux / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4 environment.
set -euo pipefail
cd "$(dirname "$0")"
ENV_NAME="${1:-tgs}"
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash setup.sh [conda-environment-name] (default: tgs)"
    exit 0
fi
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "The reference CUDA environment requires Linux; use requirements.txt for a custom environment." >&2
    exit 1
fi
command -v conda >/dev/null || { echo "Install Conda before running setup.sh." >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.12 -y
conda activate "${ENV_NAME}"
python -m pip install -r requirements.txt
python -m pip install torch_geometric==2.6.1
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    --only-binary=:all: --find-links https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install einops gpustat plotly wandb sentence-transformers
printf 'Environment ready. Activate with: conda activate %s\n' "${ENV_NAME}"
