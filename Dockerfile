FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY static ./static

RUN mkdir -p /app/data/output /app/data/cache

EXPOSE 8000
CMD ["uvicorn", "secretary_emillia.api:app", "--host", "0.0.0.0", "--port", "8000"]

