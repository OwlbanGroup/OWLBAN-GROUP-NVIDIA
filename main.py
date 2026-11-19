"""Compatibility entrypoint for running the production OWLBAN app."""

from app.main import app as app


if __name__ == "__main__":
    import uvicorn
    from app.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
