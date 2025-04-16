import logging
from logging.config import dictConfig
from typing import Dict, Any

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "app": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Application settings
APP_SETTINGS = {
    "title": "OCR Processing Service",
    "description": "Service for OCR processing with optional features",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# API settings
API_SETTINGS = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 8,
}

# CORS settings
CORS_SETTINGS = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

def setup_logging() -> logging.Logger:
    """Configure logging and return the application logger"""
    dictConfig(LOGGING_CONFIG)
    return logging.getLogger("app") 