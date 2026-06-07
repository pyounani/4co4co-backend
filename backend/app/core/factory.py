from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.v1.routers import api_router
from app.core.config.cors import setup_cors
from app.core.config.openapi import custom_openapi
from app.core.config.settings import settings
from app.core.db.database import lifespan
from app.core.exceptions.handlers import (
    validation_exception_handler,
    app_error_handler,
    generic_exception_handler
)
from app.core.exceptions.types import AppError
from app.core.middleware import ProcessTimeMiddleware


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    # 미들웨어 등록
    app.add_middleware(ProcessTimeMiddleware)

    # CORS 설정
    setup_cors(app)

    # 라우터 등록
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # 예외 핸들러 등록
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # OpenAPI 문서
    app.openapi = lambda: custom_openapi(app)

    return app