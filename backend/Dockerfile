FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# uv 설치 (공식 Docker 이미지에서 바이너리만 복사)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 의존성 파일 복사
COPY pyproject.toml uv.lock ./

# 시스템 Python에 직접 설치 (venv 불필요), dev 그룹 제외
RUN UV_SYSTEM_PYTHON=1 uv sync --frozen --no-group dev --no-install-project

# 앱 코드 복사
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
