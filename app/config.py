"""Application configuration."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "app" / "static"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
LEGACY_STATIC_DIR = BASE_DIR / "static"
DATASET_DIR = BASE_DIR / "dataset"


class Config:
    """Base configuration shared across environments."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me-in-production")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 64 * 1024))
    MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 10_000))
    MIN_TEXT_LENGTH = int(os.environ.get("MIN_TEXT_LENGTH", 20))

    MODEL_DIR = MODEL_DIR
    LEGACY_STATIC_DIR = LEGACY_STATIC_DIR

    JSON_SORT_KEYS = False
    PROPAGATE_EXCEPTIONS = False

    RATELIMIT_DEFAULT = os.environ.get("RATELIMIT_DEFAULT", "60 per minute")
    RATELIMIT_PREDICT = os.environ.get("RATELIMIT_PREDICT", "20 per minute")
    RATELIMIT_VERIFY = os.environ.get("RATELIMIT_VERIFY", "10 per minute")
    RATELIMIT_STORAGE_URI = os.environ.get("RATELIMIT_STORAGE_URI", "memory://")

    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

    # Live fact-checking (web search + LLM reasoning over retrieved sources).
    # Separate from the offline TF-IDF classifier — unset means the /verify
    # endpoint responds 503 rather than silently degrading.
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    FACTCHECK_MAX_SOURCES = int(os.environ.get("FACTCHECK_MAX_SOURCES", 5))


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    DEBUG = False
    ENV = "production"

    @classmethod
    def validate(cls) -> None:
        if cls.SECRET_KEY == "dev-change-me-in-production":
            raise RuntimeError(
                "SECRET_KEY must be set to a strong random value in production."
            )


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    RATELIMIT_ENABLED = False
    WTF_CSRF_ENABLED = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}


def get_config(name: str | None = None) -> type[Config]:
    env = name or os.environ.get("FLASK_ENV", "development")
    return config_by_name.get(env, DevelopmentConfig)
