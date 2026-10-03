#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
exec "$project_root/.qwen-wsl-venv/bin/python" "$project_root/scripts/qwen_worker.py"
