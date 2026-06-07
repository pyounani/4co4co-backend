# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

4co4co-backend is an async REST API that accepts images, uploads them to S3, and orchestrates AI-driven background music generation via an external AI server. Results are streamed to clients in real time via SSE backed by Redis Pub/Sub.

## Commands

**Install dependencies**
```bash
uv sync
```

**Run locally (requires .env)**
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop asyncio
```

**Run full stack (Redis + MongoDB + API + Celery worker)**
```bash
docker-compose up --build
```

**Run Celery worker alone**
```bash
celery -A app.core.tasks.celery_app.celery_app worker --loglevel=info --concurrency=1 --pool=prefork --prefetch-multiplier=1 -Q celery
```

**Lint / format**
```bash
uv run black app/
uv run isort app/
uv run mypy app/
```

**Tests**
```bash
uv run pytest
```

## Environment Variables (`.env`)

Required:
- `MONGO_URI`, `MONGO_DB`, `MONGO_INITDB_DATABASE`
- `REDIS_URL`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_S3_BUCKET_NAME`
- `AI_SERVER_URL`
- `API_PREFIX` (e.g. `/api/v1`)

Optional:
- `APP_ENV` — `production` (default) or `development`; only `development` exposes debug info in error responses
- `CORS_ORIGINS` — comma-separated origins, default `http://localhost:5173`
- `SSE_TIMEOUT` — seconds before SSE connection closes, default `180`
- `DISCORD_WEBHOOK_URL` — error alerting webhook
- `BACKEND_PORT` — Docker port, default `8000`

## Architecture

```
app/
├── main.py                    # FastAPI app setup (CORS, exception handlers, router)
├── api/v1/
│   ├── routers.py             # Aggregates all v1 routers
│   └── lantern_api.py         # All lantern endpoints + SSE generator
├── services/
│   ├── lantern_service.py     # S3 upload, lantern creation, Celery task dispatch
│   └── music_service.py       # HTTP call to AI_SERVER_URL
├── repositories/
│   └── lantern_repository.py  # MongoDB CRUD & aggregation (Motor)
├── schemas/
│   ├── db/lantern.py          # LanternDBModel (source of truth for DB shape)
│   └── response/              # Pydantic response models
└── core/
    ├── config/settings.py     # Pydantic BaseSettings loaded from .env
    ├── db/database.py         # Motor client, lifespan context, get_mongo_client()
    ├── tasks/
    │   ├── celery_app.py      # Celery + Redis broker configuration
    │   └── music_tasks.py     # process_lantern_music Celery task (retry logic, Redis publish)
    ├── exceptions/            # AppError hierarchy + FastAPI exception handlers
    ├── logging/logger.py      # Structured logging; Discord webhook for ERROR level
    ├── response/response.py   # success_response / success_no_cache_response helpers
    └── validation/            # name & image validation logic
```

### Request flow

1. `POST /api/v1/lanterns` — validates input, checks Redis queue length (rate limit: max 50), uploads images to S3, inserts `LanternDBModel` to MongoDB, enqueues one `process_lantern_music` Celery task per image.
2. **Celery worker** (`music_tasks.py`) — calls `AI_SERVER_URL`, stores the resulting audio in S3, updates `music_statuses` in MongoDB, publishes the result to Redis channel `lantern_music:{lantern_id}`.
3. `GET /api/v1/lanterns/{lantern_id}/music-status` — SSE endpoint; subscribes to `lantern_music:{lantern_id}` and streams events until all 3 music tracks finish or `SSE_TIMEOUT` elapses. Reconnect is supported via `last-event-id` + `?resume=true`.

### Lantern ID format

`{name}-{4-digit-random}` (e.g. `홍길동-4237`). Validated by regex `^[가-힣a-zA-Z0-9]+-[0-9]{4}$` on path/query params.

### Key domain model — `LanternDBModel`

Fields: `lantern_id`, `user_name`, `images` (list of `ImageInfo`), `music_statuses` (list of `MusicStatusInfo`: `pending | success | failed`), `musics` (list of `MusicInfo` with S3 key + timestamp), `is_public`, `created_at`.

### MongoDB

Accessed via `get_mongo_client(request)` (injected from `request.state.mongo_client`). The client is opened once at startup via FastAPI's `lifespan`. Collection: `lantern`, unique index on `lantern_id`.

### Celery

Broker and result backend both use Redis. Visibility timeout: 5400 s. Task soft limit: 40 min, hard limit: 60 min. Single worker, `prefetch-multiplier=1` for even task distribution.
