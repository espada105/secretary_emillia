#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
rvc_python="$project_root/.rvc-gpu-wsl-venv/bin/python"
if [[ ! -x "$rvc_python" ]]; then
  echo "GPU RVC 환경을 사용할 수 없습니다. CPU 환경으로 되돌립니다." >&2
  rvc_python="$project_root/.rvc-wsl-venv/bin/python"
fi
ffmpeg_binary="$($rvc_python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
mkdir -p /tmp/secretary-rvc-bin
ln -sf "$ffmpeg_binary" /tmp/secretary-rvc-bin/ffmpeg

exec env PATH="/tmp/secretary-rvc-bin:/usr/local/bin:/usr/bin:/bin" \
  bash -c 'cd "$1/vendor/rvc" && shift && exec "$@"' bash "$project_root" \
  "$rvc_python" -m infer.cli "$@" --f0-method rmvpe --format wav --overwrite
