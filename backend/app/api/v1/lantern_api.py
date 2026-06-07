import json
import time
from typing import Optional, List, Set, Dict, Any

import aioredis
from fastapi import APIRouter, UploadFile, File, Form, Query, Path, Request, Header, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.config.settings import settings
from app.core.db.database import get_mongo_client
from app.core.logging.logger import get_logger
from app.core.response import success_response, success_no_cache_response
from app.core.validation.lantern_validation import validate_name, validate_images
from app.repositories.lantern_repository import LanternRepository
from app.schemas.response.lantern_detail_response import LanternDetailResponseModel
from app.schemas.response.lantern_response import LanternResponseModel
from app.schemas.response.schemas import ResponseModel
from app.schemas.swagger import (
    error_400, error_404, error_500,
    error_400_lantern_examples, success_200_create_lantern
)
from app.services.lantern_service import LanternService

logger = get_logger(__name__)
router = APIRouter()


# --- SSE Helper ---

def get_initial_sent_tasks(doc: Dict[str, Any], resume: bool, last_event_id: Optional[str]) -> Set[str]:
    """재연결 시 last_event_id가 있는 경우 -> 이미 보낸 것을 체크"""
    sent_task_ids = set()
    if resume and last_event_id:
        statuses = doc.get("music_statuses", [])
        for s in statuses:
            sent_task_ids.add(s["task_id"])
            if s["task_id"] == last_event_id:
                break
    return sent_task_ids


def get_new_status_updates(statuses: List[Dict[str, Any]], sent_task_ids: Set[str]) -> List[Dict[str, Any]]:
    """DB에 이미 기록된 상태 중 아직 전송하지 않은 업데이트 추출"""
    return [
        s for s in statuses
        if s["status"] in ["success", "failed"] and s["task_id"] not in sent_task_ids
    ]


def format_sse_event(status_data: Dict[str, Any]) -> Dict[str, str]:
    """데이터를 SSE 포맷으로 변환"""
    event_name = "music_done_partial" if status_data["status"] == "success" else "music_failed"
    return {
        "id": status_data["task_id"],
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
    """실시간 이벤트를 생성하는 비동기 제너레이터"""
    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"lantern_music:{lantern_id}")

    start_time = time.time()

    try:
        # 잘못된 요청 (lantern 을 생성한 적이 없는 lantern_id)
        doc = await repo.find_by_lantern_id(lantern_id)
        if not doc:
            yield {"event": "error", "data": "Lantern not found"}
            return

        sent_task_ids = get_initial_sent_tasks(doc, resume, last_event_id)
        initial_updates = get_new_status_updates(doc.get("music_statuses", []), sent_task_ids)

        for s in initial_updates:
            yield format_sse_event(s)
            sent_task_ids.add(s["task_id"])

        # 2. 실시간 루프 (Redis Pub/Sub 대기)
        while not await request.is_disconnected():
            # 타임아웃 체크 (서버 자원 보호)
            if time.time() - start_time > settings.SSE_TIMEOUT:
                logger.info(f"[SSE] Timeout reached for {lantern_id}")
                break

            # Redis 메시지 수신 (비동기)
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)

            if message:
                data = json.loads(message['data'])
                if data["task_id"] not in sent_task_ids:
                    yield format_sse_event(data)
                    sent_task_ids.add(data["task_id"])

                # 모든 작업 완료 여부 확인 -> 3개 배경음 완성이 되었는지 (메시지가 올 경우에만)
                current_doc = await repo.find_by_lantern_id(lantern_id)
                statuses = current_doc.get("music_statuses", [])

                if all(s["status"] in ["success", "failed"] for s in statuses): # 진행 중인 것이 없는 것을 확인
                    yield {
                        "id": "done-all",
                        "event": "music_done_all",
                        "data": json.dumps(statuses)
                    }
                    break

    except Exception as e:
        logger.error(f"[SSE Error] {e}", exc_info=True)
        yield {"event": "error", "data": str(e)}
    finally:
        await pubsub.unsubscribe(f"lantern_music:{lantern_id}")
        await redis.close()
        logger.info(f"[SSE] Connection closed for {lantern_id}")


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
        ping=15
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
        # aioredis를 사용하여 비동기로 큐 길이 확인
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
        # Redis가 죽어있는 경우
        logger.error(f"[Redis Dead] 연결 실패: {e}")
        raise HTTPException(
            status_code=503,
            detail="서비스 통신이 원활하지 않습니다. 잠시 후 다시 시도해주세요."
        )
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        logger.error(f"[Unknown Error] {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    # 2. Validation & Processing
    db = get_mongo_client(request)
    validate_name(name)
    validate_images(images)

    lantern_service = LanternService(db)
    lantern_id = await lantern_service.create_lanterns(
        name=name, images=images, is_public=is_public
    )

    logger.info(f"[create_lanterns] Lantern created: {lantern_id}, waiting_count={queue_length}")

    # 응답에 현재 대기 작업 수를 포함하여 전달
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