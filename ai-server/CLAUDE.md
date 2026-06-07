# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the server locally
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Build and run via Docker (CPU)
docker-compose up --build

# Build for GPU (edit docker-compose.yml to use GPU service)
docker build --build-arg USE_GPU=true --build-arg BASE_IMAGE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime -t ai-server-gpu .
```

There are no automated tests in this project.

## Environment

Required `.env` variables (see `app/utils/settings.py`):
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
AWS_S3_BUCKET_NAME=
APP_ENV=development         # optional, default: production
LOG_LEVEL=INFO              # optional
DISCORD_WEBHOOK_URL=        # optional, for error alerts
```

## Architecture

This is a FastAPI AI server (port 8001) that generates background music from images using a multi-model ML pipeline.

### Request Flow

`POST /api/v1/generate-music` → `app/api/v1/generation.py` → `AIEngine.generate()` → S3 upload → response

1. Download input image from S3 (`body.image_path` is an S3 key)
2. Run emotion analysis pipeline → returns `{ emotion, caption }`
3. Build text prompt from emotion + caption
4. Generate audio with MusicGen
5. Upload generated `.mp3` to S3 at `generated/<uuid>.mp3`
6. Free GPU memory and return metadata

### Emotion Analysis Pipeline (`app/emotion/main_library.py`)

Five models run sequentially; results are weighted-averaged into a single emotion label from 17 classes (Happiness, Confidence, Fear, etc.):

| Model | Role | Weight (person) | Weight (no person) |
|---|---|---|---|
| YOLO (YOLOv8s) | Person detection | — | — |
| EMOTIC / Face | Facial emotion | 0.5 | 0 |
| CLIP MLP | Caption-based emotion | 0.3 | 0.4 |
| Color K-means | Color-based emotion | 0.2 | 0.6 |
| Moondream2 | Image captioning (feeds CLIP) | — | — |

`_MODELS` dict in `main_library.py` is the global model registry shared across the codebase.

### Music Generation

Uses `facebook/musicgen-small` via Hugging Face Transformers. MusicGen is loaded permanently at startup onto GPU. Prompt format: `"music expressing {emotion} emotion, inspired by: {caption}"`.

### GPU Memory Strategy

- All models load at startup (`@app.on_event("startup")`)
- After each request, `AIEngine.unload_models()` moves MusicGen back to CPU and calls `torch.cuda.empty_cache()`
- A semaphore (`gpu_semaphore = Semaphore(1)`) in `music_service.py` serializes GPU access

### Required Model Files

Place local model files at these paths before running:
- `model/yolo/yolov8s.pt`
- `model/clip/mlp.pt`
- `model/emotic/` (EMOTIC face model weights)
- `facebook/musicgen-small` is auto-downloaded from HuggingFace Hub

### Dead Code / Duplicate Implementations

There are two parallel implementations that are NOT both active:
- **Active**: `app/api/v1/generation.py` + `app/services/ai_engine.py` (sync, registered in `main.py`)
- **Inactive**: `app/api/v1/music_api.py` + `app/services/music_service.py` (async, not registered)

`generation.py` also imports `from app.services.storage import StorageService` which does not exist — this will raise an `ImportError` at runtime. S3 operations are actually in `app/utils/s3.py`.

### Key Files

| File | Purpose |
|---|---|
| `main.py` | App entry, model preload on startup |
| `app/api/v1/generation.py` | Single endpoint `POST /api/v1/generate-music` |
| `app/emotion/main_library.py` | `_MODELS` registry, full emotion pipeline, MusicGen |
| `app/services/ai_engine.py` | `AIEngine` class wrapping inference + memory lifecycle |
| `app/utils/s3.py` | Async S3 helpers (upload, download, presigned URL) |
| `app/utils/settings.py` | Pydantic Settings — all env var definitions |
| `app/utils/logger.py` | Structured logger with optional Discord webhook on ERROR |
