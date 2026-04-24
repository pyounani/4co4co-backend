import time
import json
import requests
import redis  # pip install redis 필요
from datetime import datetime
from celery import shared_task

from app.core.config.settings import settings
from app.core.logging.logger import get_logger
from app.core.db.database import get_mongo_sync_client

logger = get_logger(__name__)

# [설정] Celery용 Sync Redis 클라이언트 (브로커 URL과 동일하게 설정)
redis_client = redis.from_url(settings.CELERY_BROKER_URL)


def publish_status(lantern_id: str, image_key: str, status: str, message: str, s3_key: str = None):
    """
    Redis 채널에 실시간 상태를 발행(Publish)합니다.
    DB에는 저장하지 않습니다. (휘발성 메시지)
    """
    channel = f"lantern_status:{lantern_id}"
    payload = {
        "image_s3": image_key,
        "status": status,
        "message": message,
        "s3_key": s3_key,
        "timestamp": str(datetime.utcnow())
    }
    # Redis로 메시지 쏘기!
    redis_client.publish(channel, json.dumps(payload))


def update_db_result(collection, lantern_id: str, image_key: str, status: str, s3_key: str = None,
                     error_msg: str = None):
    """
    [중요] 최종 결과(성공/실패)만 DB에 영구 저장합니다.
    """
    update_fields = {
        "music_statuses.$.status": status,
        "music_statuses.$.updated_at": datetime.utcnow()
    }

    if s3_key:
        update_fields["music_statuses.$.s3_key"] = s3_key
        # 성공 시에는 musics 배열에도 추가
        push_fields = {
            "musics": {
                "s3_path": s3_key,
                "created_at": datetime.utcnow()
            }
        }
    else:
        push_fields = None

    if error_msg:
        update_fields["music_statuses.$.error_msg"] = error_msg

    try:
        query = {"lantern_id": lantern_id, "music_statuses.image_s3": image_key}
        update_op = {"$set": update_fields}
        if push_fields:
            update_op["$push"] = push_fields

        collection.update_one(query, update_op)
    except Exception as e:
        logger.error(f"[DB Update Error] {lantern_id}: {e}")


def call_ai_server_sync(image_key: str) -> str:
    url = f"{settings.AI_SERVER_URL}{settings.API_PREFIX}/generate-music"
    payload = {"image_path": image_key}
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "success":
        raise ValueError(f"AI Server Error: {body.get('message')}")
    s3_key = body.get("data", {}).get("s3_key")
    if not s3_key:
        raise ValueError("Invalid s3_key")
    return s3_key


@shared_task(
    name="process_lantern_music",
    bind=True,
    acks_late=True,
    retry_kwargs={"max_retries": 3}
)
def process_lantern_music(self, lantern_id: str, image_key: str):
    start_time = time.time()
    logger.info(f"[Task Start] lantern_id={lantern_id}")

    db = get_mongo_sync_client()
    collection = db["lanterns"]

    # [Step 1] 작업 시작 (Redis 알림만!) -> DB 저장 X
    publish_status(lantern_id, image_key, "processing", "이미지 분위기 분석 중...")

    try:
        # AI Server Call
        s3_key = call_ai_server_sync(image_key)

        # [Step 3] 성공 (DB 저장 O + Redis 알림 O)
        # 1. DB에 영구 저장
        update_db_result(collection, lantern_id, image_key, "success", s3_key=s3_key)
        # 2. 사용자에게 알림
        publish_status(lantern_id, image_key, "success", "완성되었습니다!", s3_key=s3_key)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[Task Success] {lantern_id} ({elapsed}s)")
        return s3_key

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        retries = self.request.retries
        max_retries = self.retry_kwargs.get('max_retries', 3)

        if retries < max_retries:
            # [Step 2] 재시도 (Redis 알림만!) -> DB 저장 X
            wait_seconds = 2 ** retries
            msg = f"사용자가 많아 대기 중입니다... ({retries + 1}/{max_retries})"

            publish_status(lantern_id, image_key, "retrying", msg)

            raise self.retry(exc=e, countdown=wait_seconds)

        else:
            # [Step 4] 최종 실패 (DB 저장 O + Redis 알림 O)
            update_db_result(collection, lantern_id, image_key, "failed", error_msg=str(e))
            publish_status(lantern_id, image_key, "failed", "생성에 실패했습니다.")
            raise e