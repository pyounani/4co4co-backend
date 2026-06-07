from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.middleware.cors import CORSMiddleware

from app.api.v1.routers import api_router
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

    app.add_middleware(ProcessTimeMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_PREFIX)

    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.openapi = lambda: custom_openapi(app)

    return app