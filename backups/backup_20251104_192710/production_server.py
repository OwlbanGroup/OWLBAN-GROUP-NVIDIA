#!/usr/bin/env python3
"""
Production WSGI server for JPMorgan Financial APIs
"""
import os
import logging
from waitress import serve
from app_final import app

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def configure_production_app():
    """Configure the Flask app for production"""
    # Set production environment variables
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('SECRET_KEY', os.environ.get('SECRET_KEY', 'production-secret-key-change-in-env'))

    # Configure Flask app for production
    app.config.update(
        TESTING=False,
        DEBUG=False,
        SECRET_KEY=os.environ.get('SECRET_KEY'),
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        PERMANENT_SESSION_LIFETIME=3600,  # 1 hour
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB
        JSON_SORT_KEYS=False,
        JSONIFY_PRETTYPRINT_REGULAR=False
    )

    # Configure rate limiting for production (Redis if available)
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address

        # Try to use Redis for rate limiting in production
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            limiter = Limiter(
                app=app,
                key_func=get_remote_address,
                storage_uri=redis_url,
                storage_options={"socket_connect_timeout": 30},
                strategy="fixed-window"
            )
            logger.info("✅ Redis-based rate limiting configured")
        else:
            # Fallback to in-memory (not recommended for production)
            limiter = Limiter(
                app=app,
                key_func=get_remote_address,
                storage_uri="memory://",
                strategy="fixed-window"
            )
            logger.warning("⚠️ Using in-memory rate limiting (not recommended for production)")

    except Exception as e:
        logger.error(f"Failed to configure rate limiting: {e}")

    return app

if __name__ == "__main__":
    logger.info("🚀 Starting JPMorgan Financial APIs Production Server")

    # Configure the app for production
    app = configure_production_app()

    # Production server configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8000))
    workers = int(os.environ.get('WORKERS', 4))

    logger.info(f"📍 Server will be available at: http://{host}:{port}")
    logger.info(f"🔧 Using Waitress WSGI server with {workers} threads")
    logger.info("🏭 Production configuration applied")

    # Start the production server
    try:
        serve(
            app,
            host=host,
            port=port,
            threads=workers,
            url_prefix='/',
            channel_timeout=300,
            cleanup_interval=30,
            max_request_body_size=104857600,  # 100MB
            max_request_header_size=8190,
            inbuf_overflow=104857600,  # 100MB
            outbuf_overflow=104857600,  # 100MB
            connection_limit=1000,
            backlog=2048,
            # Security headers
            ident='JPMorgan Financial APIs Production Server'
        )
    except Exception as e:
        logger.error(f"Failed to start production server: {e}")
        raise
