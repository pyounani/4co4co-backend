import asyncio
import io
from typing import Optional, Tuple
from uuid import uuid4

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config.settings import settings
from app.core.logging.logger import get_logger

logger = get_logger(__name__)


class S3Service:

    def __init__(self):
        self.session = aioboto3.Session()
        self._s3_client = None
        self._client_cm = None
        self._lock = asyncio.Lock()

    async def get_s3_client(self):

        if self._s3_client:
            return self._s3_client

        async with self._lock:
            if self._s3_client is None:
                self._client_cm = self.session.client(
                    "s3",
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_REGION,
                    config=Config(retries={"max_attempts": 3, "mode": "adaptive"})
                )
                self._s3_client = await self._client_cm.__aenter__()
                logger.info("[S3Service] S3 client initialized")

        return self._s3_client

    async def close(self):

        if self._client_cm:
            await self._client_cm.__aexit__(None, None, None)
            self._client_cm = None
            self._s3_client = None
            logger.info("[S3Service] S3 client closed")


s3_service = S3Service()


async def upload_file_to_s3(file: UploadFile, folder: str = "uploads") -> Tuple[Optional[str], int]:

    try:
        try:
            file_obj = file.file
            pos = file_obj.tell()
            file_obj.seek(0, io.SEEK_END)
            file_size = file_obj.tell()
            file_obj.seek(0)
        except Exception:
            file_size = 0
            logger.warning(f"[S3 Upload] Failed to calculate file size: {file.filename}")

        safe_folder = (folder or "uploads").strip("/")
        file_extension = file.filename.split(".")[-1] if file.filename and "." in file.filename else ""
        s3_file_key = f"{safe_folder}/{uuid4()}{('.' + file_extension) if file_extension else ''}"

        logger.info(f"[S3 Upload] Start: filename={file.filename}, size={file_size}, key={s3_file_key}")

        s3_client = await s3_service.get_s3_client()

        await s3_client.upload_fileobj(
            file.file,
            settings.AWS_S3_BUCKET_NAME,
            s3_file_key,
            ExtraArgs={
                "ContentType": file.content_type or "application/octet-stream",
                "Metadata": {
                    "original_filename": file.filename or "unknown",
                    "file_size": str(file_size)
                }
            }
        )

        logger.info(f"[S3 Upload] Success: key={s3_file_key}, bucket={settings.AWS_S3_BUCKET_NAME}")
        return s3_file_key, file_size

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"[S3 Upload] AWS Error: filename={file.filename}, error_code={error_code}, error={str(e)}")
        return None, 0

    except Exception as e:
        logger.exception(f"[S3 Upload] Unexpected Error: filename={file.filename}, error={str(e)}")
        return None, 0

    finally:
        file.file.seek(0)


async def delete_file_from_s3(s3_key: str) -> bool:

    try:
        s3_client = await s3_service.get_s3_client()
        await s3_client.delete_object(
            Bucket=settings.AWS_S3_BUCKET_NAME,
            Key=s3_key
        )
        logger.info(f"[S3 Delete] Success: key={s3_key}, bucket={settings.AWS_S3_BUCKET_NAME}")
        return True

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"[S3 Delete] AWS Error: key={s3_key}, error_code={error_code}, error={str(e)}")
        return False

    except Exception as e:
        logger.exception(f"[S3 Delete] Unexpected Error: key={s3_key}, error={str(e)}")
        return False


async def get_file_url_from_s3(s3_key: str, expires_in: int = 3600) -> Optional[str]:

    try:
        s3_client = await s3_service.get_s3_client()
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': settings.AWS_S3_BUCKET_NAME,
                'Key': s3_key
            },
            ExpiresIn=expires_in
        )
        logger.info(f"[S3 URL] Generated: key={s3_key}, expires_in={expires_in}")
        return url

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"[S3 URL] AWS Error: key={s3_key}, error_code={error_code}, error={str(e)}")
        return None

    except Exception as e:
        logger.exception(f"[S3 URL] Unexpected Error: key={s3_key}, error={str(e)}")
        return None


async def init_s3_service():

    logger.info("[S3 Service] Initializing...")


async def close_s3_service():

    logger.info("[S3 Service] Closing connections...")
    await s3_service.close()


# --- Presigned URL ---

async def generate_presigned_url(s3_key: str, expires_in: int = 3600) -> Optional[str]:

    try:
        s3_client = await s3_service.get_s3_client()

        if asyncio.iscoroutinefunction(s3_client.generate_presigned_url):
            url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.AWS_S3_BUCKET_NAME, "Key": s3_key},
                ExpiresIn=expires_in,
            )
        else:
            url = await run_in_threadpool(
                s3_client.generate_presigned_url,
                "get_object",
                Params={"Bucket": settings.AWS_S3_BUCKET_NAME, "Key": s3_key},
                ExpiresIn=expires_in,
            )

        logger.info(f"[S3 Presigned URL] Generated: key={s3_key}, expires_in={expires_in}")
        return url
    except Exception as e:
        logger.exception(f"[S3 Presigned URL] Failed: {e}")
        return None
