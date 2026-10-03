# Secretary Emillia

개인 로컬 비서의 첫 단계인 한국어 TTS 테스트 앱입니다. FastAPI는 `localhost`에서만 노출됩니다.

```powershell
Copy-Item .env.example .env
docker compose up --build
```

브라우저에서 `http://localhost:8000`을 열고 문장을 입력하면 WAV가 생성·재생됩니다. 생성 파일은 `data/output/`에만 남으며 Git에서 제외됩니다.

## 음성 실험 페이지

기본값은 다음의 순차 로컬 파이프라인입니다.

`Qwen3-TTS 1.7B (말투·감정) → RVC (Emilia 또는 Ram 음색)`

페이지에서 말투 지시문, 캐릭터, 피치, 음색 강도(`index rate`), 자음 보호(`protect`)를 조절할 수 있습니다. RVC 음질이 깨지면 음색 강도를 `0.35~0.50`으로 낮추고 자음 보호를 `0.40~0.50`으로 올려 보세요.

TTS 엔진 선택에는 다음 두 Qwen 모델이 있습니다.

- `Qwen3 1.7B CustomVoice`: Sohee 한국어 화자와 말투 지시문을 사용합니다.
- `Qwen3 0.6B Korean`: `daje/Qwen3-TTS-12Hz-0.6B-Korean`의 `korean_tts` 화자를 사용합니다. 한국어 발음 비교용이며 말투 지시는 지원하지 않습니다.

Qwen/RVC는 Windows 네이티브 패키지 충돌을 피하기 위해 WSL의 별도 `uv` 가상환경에서 실행됩니다. 다음 경로가 준비돼 있어야 합니다.

- `.qwen-wsl-venv/`: `qwen-tts`, CUDA PyTorch, Qwen3-TTS 1.7B 가중치
- `.rvc-gpu-wsl-venv/`: RTX 50 계열용 CUDA RVC 추론 런타임 (준비되면 자동 사용)
- `.rvc-wsl-venv/`: GPU 런타임이 없을 때만 쓰는 CPU RVC 폴백
- `models/rezero-lim/extracted/`: Emilia/Ram `.pth`와 `.index` 파일

Qwen 워커는 FastAPI가 살아 있는 동안 마지막으로 사용한 Qwen 모델을 GPU에 유지합니다. 따라서 첫 생성 또는 엔진 변경 때만 모델 로딩 시간이 들고, 같은 엔진을 연속으로 쓰는 요청은 바로 추론을 시작합니다. RVC는 현재 요청마다 CPU 추론을 실행하므로 Emilia/Ram 경로의 남은 지연 시간은 별도로 존재합니다.

`POST /api/tts`에는 `speak` 불리언이 있습니다. `true`면 WAV를 생성하고, `false`면 음성을 만들지 않고 `{"spoken": false}`를 반환합니다. 이후 음성 입력 명령이나 에이전트 응답에서, 응답을 읽을지 여부를 이 값으로 제어할 수 있습니다.

`base_korean`을 선택하면 RVC 없이 Qwen의 한국어 기본 화자 Sohee를 사용합니다. MeloTTS는 API 호환성 확인을 위한 `melo` 엔진으로 남아 있으며, 현재 UI의 기본 경로는 아닙니다.

Docker Compose는 기본 FastAPI/MeloTTS 환경을 고정합니다. Qwen/RVC까지 컨테이너에서 사용하려면 GPU 지원 Docker 및 모델 볼륨을 별도로 구성해야 하므로, 현재 검증된 경로는 Windows 호스트 FastAPI + WSL 추론입니다.

RTX 50 계열 GPU RVC 환경은 WSL에서 한 번만 준비합니다.

```bash
bash scripts/setup_rvc_gpu_wsl.sh
```

이 환경은 CUDA 12.8 PyTorch를 사용합니다. GPU RVC 검증에서 Emilia 변환이 `cuda:0`으로 실행됐으며, 예시 음성의 추론 시간은 특징 추출 0.67초, RMVPE 피치 추정 20.68초, 합성 1.33초였습니다. 첫 요청은 모델 초기화 때문에 더 오래 걸립니다.

```powershell
docker compose down
```
