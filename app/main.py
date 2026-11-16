"""Runtime entrypoint for the production-ready OWLBAN application."""

from __future__ import annotations

import uvicorn

from app.app_factory import app as app
from app.config import get_settings


settings = get_settings()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
