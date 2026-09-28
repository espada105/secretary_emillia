#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
rvc_python="$project_root/.rvc-wsl-venv/bin/python"
ffmpeg_binary="$($rvc_python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p /tmp/secretary-rvc-bin
ln -sf "$ffmpeg_binary" /tmp/secretary-rvc-bin/ffmpeg

exec env PATH="/tmp/secretary-rvc-bin:/usr/local/bin:/usr/bin:/bin" \
  bash -c 'cd "$1/vendor/rvc" && shift && exec "$@"' bash "$project_root" \
  "$rvc_python" -m infer.cli "$@" --f0-method rmvpe --format wav --overwrite
