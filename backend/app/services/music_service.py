from datetime import datetime

import httpx

from app.core.config.settings import settings
from app.core.exceptions.types import AppError, NotFoundError
from app.repositories.lantern_repository import LanternRepository
from app.schemas.db.lantern import MusicInfo
from app.core.logging.logger import get_logger

logger = get_logger(__name__)


def call_ai_server(image: str) -> str:

    url = f"{settings.AI_SERVER_URL}{settings.API_PREFIX}/generate-music"
    payload = {"image_path": image}

    try:
        logger.info(f"[AI Server] Request: url={url}, payload={payload}")
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()
        logger.info(f"[AI Server] Response: status_code={resp.status_code}, body={body}")
    except Exception as e:
        logger.error(f"[AI Server] Request failed: {e}", exc_info=True)
        raise AppError(f"AI server request failed: {e}")

    # 응답 Validate
    if body.get("status") != "success":
        logger.error(f"[AI Server] Returned error: {body.get('message', 'unknown')}")
        raise AppError(f"AI returned error: {body.get('message', 'unknown')}")

    s3_key = body.get("data", {}).get("s3_key")
    if not s3_key or not isinstance(s3_key, str):
        logger.error("[AI Server] Invalid s3_key in response")
        raise AppError("AI server did not return a valid s3_key")

    logger.info(f"[AI Server] Music generated successfully: s3_key={s3_key}")
    return s3_key


class MusicService:

    def __init__(self, db):
        self.db = db
        self.lantern_repo = LanternRepository(db)

    """
    AI 서버로 요청보내기
    """
    def generate_music(
        self,
        lantern_id: str,
        image: str,
    ) -> str:

        logger.info(f"[MusicService] Generate music start: lantern_id={lantern_id}, image={image}")

        lantern = self.lantern_repo.collection.find_one({"lantern_id": lantern_id})
        if not lantern:
            logger.warning(f"[MusicService] Lantern not found: {lantern_id}")
            raise NotFoundError(f"Lantern ID {lantern_id} not found.")

        s3_key = call_ai_server(image)

        music_info = MusicInfo(
            s3_path=s3_key,
            created_at=datetime.utcnow()
        ).model_dump(exclude={"id"})

        self.lantern_repo.collection.update_one(
            {"lantern_id": lantern_id},
            {"$push": {"musics": music_info}}
        )

        logger.info(f"[MusicService] Music info saved: lantern_id={lantern_id}, s3_key={s3_key}")
        return s3_key