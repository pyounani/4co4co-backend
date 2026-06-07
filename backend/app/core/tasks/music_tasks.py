import time
from app.core.db.database import get_db
from app.core.logging.logger import get_logger
from app.core.tasks.celery_app import celery_app
from app.repositories.lantern_repository import LanternRepository
from app.services.music_service import MusicService

logger = get_logger(__name__)

@celery_app.task(
    name="process_lantern_music",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3}
)
def process_lantern_music(lantern_id: str, image_key: str):
    db = get_db()
    repo = LanternRepository(db)
    service = MusicService(db)

    # 이미 성공한 경우 -> AI 요청 안 보내기 (acks_late=True -> 작업 중복 방지)
    existing_doc = repo.collection.find_one({
        "lantern_id": lantern_id,
        "music_statuses": {
            "$elemMatch": {"image_s3": image_key, "status": "success"}
        }
    }, {"music_statuses.$": 1})

    if existing_doc:
        logger.info(f"[Skip Task] Already succeeded: lantern_id={lantern_id}, image={image_key}")
        return existing_doc["music_statuses"][0].get("s3_key")

    try:
        logger.info(f"[Task Start] lantern_id={lantern_id}, image={image_key}")
        s3_key = service.generate_music(lantern_id=lantern_id, image=image_key)

        repo.collection.update_one(
            {"lantern_id": lantern_id, "music_statuses.image_s3": image_key},
            {
                "$set": {
                    "music_statuses.$.status": "success",
                    "music_statuses.$.s3_key": s3_key,
                    "music_statuses.$.updated_at": time.time()
                }
            }
        )

        logger.info(f"[Task Success] lantern_id={lantern_id}, image={image_key}")
        return s3_key

    except Exception as e:
        logger.error(f"[Task Error] lantern_id={lantern_id}, error={e}")

        repo.collection.update_one(
            {"lantern_id": lantern_id, "music_statuses.image_s3": image_key},
            {
                "$set": {
                    "music_statuses.$.status": "failed",
                    "music_statuses.$.error_msg": str(e),
                    "music_statuses.$.updated_at": time.time()
                }
            }
        )

        raise e
