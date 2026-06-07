import asyncio
import json
import time
from typing import Optional, List, Set, Dict, Any

import aioredis
from fastapi import APIRouter, UploadFile, File, Form, Query, Path, Request, Header, HTTPException
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.core.config.settings import settings
from app.core.db.database import get_mongo_client
from app.core.logging.logger import get_logger
from app.core.response import success_response, success_no_cache_response
from app.core.validation.lantern_validation import validate_name, validate_images
from app.repositories.lantern_repository import LanternRepository
from app.schemas.response.lantern import LanternDetailResponseModel, LanternResponseModel
from app.schemas.response.schemas import ResponseModel
from app.schemas.swagger import (
    error_400, error_404, error_500,
    error_400_lantern_examples, success_200_create_lantern
)
from app.services.lantern_service import LanternService

logger = get_logger(__name__)
router = APIRouter()


# --- SSE Helper ---

def format_sse_event(status_data: Dict[str, Any]) -> Dict[str, str]:
    event_name = "music_done_partial" if status_data["status"] == "success" else "music_failed"
    return {
        "id": status_data["image_s3"],
        "event": event_name,
        "data": json.dumps(status_data)
    }


async def lantern_event_generator(
        request: Request,
        lantern_id: str,
        repo: LanternRepository,
        resume: bool,
        last_event_id: Optional[str]
):
    start_time = time.time()
    sent_image_keys: Set[str] = set()

    doc = await repo.find_by_lantern_id(lantern_id)
    if not doc:
        yield {"event": "error", "data": json.dumps({"message": "Lantern not found"})}
        return

    # 재접속: last_event_id까지 sent로 마킹 -> 중복 전송 방지
    if resume and last_event_id:
        for s in doc.get("music_statuses", []):
            sent_image_keys.add(s["image_s3"])
            if s["image_s3"] == last_event_id:
                break

    yield ServerSentEvent(retry=3000)

    while not await request.is_disconnected():
        if time.time() - start_time > settings.SSE_CONNECTION_TIMEOUT:
            logger.info(f"[SSE] Timeout: {lantern_id}")
            break

        doc = await repo.find_by_lantern_id(lantern_id)
        if not doc:
            break
        statuses = doc.get("music_statuses", [])

        for s in statuses:
            if s["status"] in ("success", "failed") and s["image_s3"] not in sent_image_keys:
                yield format_sse_event(s)
                sent_image_keys.add(s["image_s3"])

        if statuses and all(s["status"] in ("success", "failed") for s in statuses):
            yield {"id": "done-all", "event": "music_done_all", "data": json.dumps(statuses)}
            break

        await asyncio.sleep(settings.SSE_POLL_INTERVAL)

    logger.info(f"[SSE] Connection closed: {lantern_id}")


# --- 주요 API들 ---

"""
배경음 생성 완료 여부 SSE 통신하기
"""
@router.get("/lanterns/{lantern_id}/music-status")
async def music_status(
        request: Request,
        lantern_id: str = Path(..., regex=r"^[가-힣a-zA-Z0-9]+-[0-9]{4}$"),
        resume: bool = Query(False),
        last_event_id: str = Header(None, convert_underscores=False)
):
    db = get_mongo_client(request)
    repo = LanternRepository(db)

    return EventSourceResponse(
        lantern_event_generator(request, lantern_id, repo, resume, last_event_id),
        ping=settings.SSE_KEEPALIVE_INTERVAL,
        headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )


"""
랜턴 생성하기
"""
@router.post(
    "/lanterns",
    response_model=ResponseModel[dict],
    responses={200: success_200_create_lantern, 400: error_400_lantern_examples, 500: error_500}
)
async def create_lanterns(
        request: Request,
        name: str = Form(..., min_length=1, max_length=50, description="User name"),
        images: List[UploadFile] = File(..., description="Image files"),
        is_public: bool = Form(True, description="Public visibility"),
):
    """
    create a new lantern with queue length check (Rate Limiting)
    """
    logger.info(f"[create_lanterns] Request received: name={name}")

    try:
        redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        queue_length = await redis.llen("celery")
        await redis.close()

        MAX_QUEUE_SIZE = 50
        if queue_length >= MAX_QUEUE_SIZE:
            logger.warning(f"[Queue Full] current_length={queue_length}, limit={MAX_QUEUE_SIZE}")
            raise HTTPException(
                status_code=503,
                detail="현재 대기 인원이 많습니다. 잠시 후 다시 시도해주세요."
            )

    except (ConnectionError, aioredis.RedisError) as e:
        logger.error(f"[Redis Dead] 연결 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="서비스 통신이 원활하지 않습니다. 잠시 후 다시 시도해주세요."
        )
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        logger.error(f"[Unknown Error] {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    db = get_mongo_client(request)
    validate_name(name)
    validate_images(images)

    lantern_service = LanternService(db)
    lantern_id = await lantern_service.create_lanterns(
        name=name, images=images, is_public=is_public
    )

    logger.info(f"[create_lanterns] Lantern created: {lantern_id}, waiting_count={queue_length}")

    return success_response(
        data={
            "lantern_id": lantern_id,
            "waiting_count": queue_length
        },
        message="Lantern Created"
    )


"""
최근 20개 랜턴 목록 조회하기
"""
@router.get(
    "/lanterns",
    response_model=ResponseModel[List[LanternResponseModel]],
    responses={200: {"description": "Lantern List"}, 400: error_400, 404: error_404, 500: error_500}
)
async def get_lantern_list(
        request: Request,
        current_lantern_id: Optional[str] = Query(None, regex=r"^[가-힣a-zA-Z0-9]+-[0-9]{4}$")
):
    db = get_mongo_client(request)
    lantern_service = LanternService(db)
    lanterns = await lantern_service.get_recent_lanterns(current_lantern_id=current_lantern_id)

    return success_no_cache_response(
        data=[lantern.model_dump() for lantern in lanterns],
        message="Lantern list"
    )

"""
lantern_id에 따른 상세 내용 조회하기
"""
@router.get(
    "/lanterns/{lantern_id}",
    response_model=ResponseModel[LanternDetailResponseModel],
    responses={200: {"description": "Lantern Detail"}, 404: error_404}
)
async def get_lantern_detail(
        request: Request,
        lantern_id: str = Path(..., regex=r"^[가-힣a-zA-Z0-9]+-[0-9]{4}$")
):
    db = get_mongo_client(request)
    lantern_service = LanternService(db)
    lantern = await lantern_service.get_lantern_detail(lantern_id)

    return success_response(data=lantern.model_dump(), message="Lantern detail")
