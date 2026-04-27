"""
Application configuration — environment-aware settings for Flask.
"""

import os


class Config:
    """Base configuration shared by all environments."""
    DEBUG = False
    TESTING = False

    # CORS: comma-separated origins from env var
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(",")

    # External APIs
    OPENWEATHER_API = os.getenv("OPENWEATHER_API") or os.getenv("OPENWEATHER_API_KEY")
    WEATHER_URL = "https://api.openweathermap.org/data/2.5"

    # Database
    DB_PATH = os.getenv("DB_PATH", "chat_memory.db")

    # Rate limits (per IP)
    RATELIMIT_DEFAULT = "60/minute"
    RATELIMIT_CHAT = "30/minute"
    RATELIMIT_WEATHER = "60/minute"
    RATELIMIT_HEALTH = "120/minute"

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "json"  # "json" or "text"

    # Input validation
    MAX_INPUT_LENGTH = 500
    MAX_SESSION_AGE_DAYS = 30

    # CSV data
    DATA_PATH = os.getenv("DATA_PATH", "data/trends.csv")

    # ML model
    ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")


class DevelopmentConfig(Config):
    """Development-specific overrides."""
    DEBUG = True
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
    LOG_FORMAT = "text"


class TestingConfig(Config):
    """Testing-specific overrides."""
    TESTING = True
    DB_PATH = ":memory:"
    LOG_LEVEL = "WARNING"


class ProductionConfig(Config):
    """Production-specific overrides."""
    LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")


def get_config():
    """Return the config class for the current environment."""
    env = os.getenv("FLASK_ENV", "development").lower()
    configs = {
        "development": DevelopmentConfig,
        "testing": TestingConfig,
        "production": ProductionConfig,
    }
    return configs.get(env, DevelopmentConfig)()
