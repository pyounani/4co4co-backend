from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # 환경 구분
    APP_ENV: str = Field("production", description="Application environment (e.g., production, development)")
    LOG_LEVEL: str = Field("INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    DISCORD_WEBHOOK_URL: str | None = Field(
        None,
        description="Discord Webhook URL for error alerts (optional)"
    )
    CORS_ORIGINS: str = "http://localhost:5173"

    # MongoDB
    MONGO_URI: str = Field(..., description="MongoDB connection URI")
    MONGO_DB: str = Field(..., description="MongoDB database name")
    MONGO_MAX_POOL_SIZE: int = Field(10, description="Maximum number of connections in the pool")
    MONGO_MIN_POOL_SIZE: int = Field(0, description="Minimum number of connections in the pool")
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = Field(
        5000,
        description="Server selection timeout in milliseconds"
    )
    MONGO_INITDB_DATABASE: str = Field(..., description="Database to initialize on startup")

    # API
    API_PREFIX: str = Field(..., description="API prefix (e.g., /api/v1)")
    AI_SERVER_URL: str = Field(..., description="Base URL of the AI server")

    # AWS
    AWS_ACCESS_KEY_ID: str = Field(..., description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(..., description="AWS secret access key")
    AWS_REGION: str = Field(..., description="AWS region")
    AWS_S3_BUCKET_NAME: str = Field(..., description="S3 bucket name")

    # Redis
    REDIS_URL: str = Field(..., description="Redis URL for Celery Broker and Pub/Sub")

    # SSE
    SSE_POLL_INTERVAL: int = Field(2, gt=0)
    SSE_CONNECTION_TIMEOUT: int = Field(150, gt=0)
    SSE_KEEPALIVE_INTERVAL: int = Field(15, gt=0)

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
