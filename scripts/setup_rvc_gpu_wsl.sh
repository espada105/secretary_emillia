#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
environment="$project_root/.rvc-gpu-wsl-venv"

uv venv "$environment" --python 3.12
uv pip install --python "$environment/bin/python" \
  torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match
uv pip install --python "$environment/bin/python" \
  -r "$project_root/vendor/rvc/requirments_cu128_py312.txt" \
  imageio-ffmpeg \
  --index-strategy unsafe-best-match
"$environment/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
