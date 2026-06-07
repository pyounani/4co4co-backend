ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ARG USE_GPU=false
ENV USE_GPU=${USE_GPU}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 libglib2.0-0 git \
    gcc g++ build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./

RUN if [ "$USE_GPU" = "true" ]; then \
        uv sync --frozen --no-dev --extra gpu ; \
    else \
        uv sync --frozen --no-dev --extra cpu ; \
    fi

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
