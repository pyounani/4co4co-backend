from celery import Celery
from app.core.config.settings import settings

celery_app = Celery(
    "backend",
    broker=settings.REDIS_URL,
    backend=settings.MONGO_URI,
    include=["app.core.tasks.music_tasks"],
)

celery_app.conf.update(
    # 기본 설정
    task_default_queue="celery",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,

    # Worker가 task를 끝낸 뒤 ACK 전송
    # Worker 장애 시 task가 broker에 재큐잉되도록 설정
    task_acks_late=True,

    # Worker 프로세스 종료/크래시 시
    # ACK 되지 않은 task를 다시 큐로 복구
    task_reject_on_worker_lost=True,

    # 이미 시작한 task는 worker 종료 전까지 유지
    # (중간 cancel 방지)
    worker_cancel_long_running_tasks_on_connection_loss=False,

    # Redis Broker 설정
    broker_transport_options={
        # ACK 없이 일정 시간 지나면 task 재큐잉
        # AI 추론 최대 수행 시간을 고려하여 여유 있게 설정
        "visibility_timeout": 5400,

        # Redis 연결 재시도 정책
        "max_retries": 3,
        "interval_start": 0,
        "interval_step": 0.5,
        "interval_max": 3,
    },

    # Retry 정책
    task_default_retry_delay=5,
    task_annotations={
        "*": {
            "max_retries": 3,
        }
    },

    # Soft timeout:
    # graceful shutdown 유도
    task_soft_time_limit=40 * 60,

    # Hard timeout:
    # 무한 실행 방지
    task_time_limit=60 * 60,

    # Worker prefetch 최소화
    # 긴 AI task에서 특정 worker 쏠림 방지
    worker_prefetch_multiplier=1,
)