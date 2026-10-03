from __future__ import annotations

import json
import queue
import subprocess
import threading
import uuid
from pathlib import Path

from .config import Settings


class PersistentQwenWorker:
    """One long-lived WSL process that keeps the active Qwen model on the GPU."""

    def __init__(self, settings: Settings, project_root: Path) -> None:
        self._settings = settings
        self._project_root = project_root
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, object]] = queue.Queue()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    @staticmethod
    def _to_wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        if len(drive) != 1:
            raise RuntimeError(f"WSL 경로로 바꿀 수 없습니다: {resolved}")
        return f"/mnt/{drive}{resolved.as_posix().split(':', maxsplit=1)[1]}"

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        script = self._to_wsl_path(self._project_root / "scripts" / "run_qwen_worker_wsl.sh")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            ["wsl.exe", "-d", self._settings.secretary_wsl_distro, "--", "bash", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )
        self._responses = queue.Queue()
        self._reader = threading.Thread(target=self._read_responses, daemon=True)
        self._reader.start()
        return self._process

    def _read_responses(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def generate(
        self,
        *,
        text: str,
        instruct: str,
        output: Path,
        model: Path,
        speaker: str,
    ) -> None:
        with self._lock:
            process = self._start()
            if process.stdin is None:
                raise RuntimeError("Qwen 워커의 입력 채널을 열 수 없습니다.")
            request_id = uuid.uuid4().hex
            payload = {
                "id": request_id,
                "text": text,
                "instruct": instruct,
                "output": self._to_wsl_path(output),
                "model": self._to_wsl_path(model),
                "speaker": speaker,
            }
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            try:
                response = self._responses.get(timeout=360)
            except queue.Empty as exc:
                raise RuntimeError("Qwen 워커 응답 시간이 초과되었습니다.") from exc
            if response.get("id") != request_id:
                raise RuntimeError("Qwen 워커 응답 순서가 올바르지 않습니다.")
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error", "Qwen 워커 생성 실패")))

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None or process.poll() is not None:
                return
            if process.stdin is not None:
                try:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
