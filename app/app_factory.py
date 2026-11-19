"""FastAPI application factory for the production-ready OWLBAN layer."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.logging_config import configure_logging
from app.schemas import HealthResponse, OptimizationRequest, OptimizationResponse, PortfolioRiskRequest
from app.security import verify_credentials
from app.services.ai_service import AIService
from app.storage import AppStore


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter for protected endpoints."""

    def __init__(self, app: FastAPI, limit_per_minute: int = 60) -> None:
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self.requests_by_ip: Dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path not in {"/ai/optimize", "/ai/risk", "/quantum/demo"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self.requests_by_ip[client_ip]
        while window and now - window[0] > 60:
            window.popleft()
        if not window:
            self.requests_by_ip.pop(client_ip, None)
            window = self.requests_by_ip[client_ip]
        if len(window) >= self.limit_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please slow down your requests."},
            )
        window.append(now)
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request IDs and emit structured request lifecycle logs."""

    async def dispatch(self, request: Request, call_next):
        import logging
        import uuid

        logger = logging.getLogger("app.request")
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                },
            )
            raise

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
            },
        )
        return response


def create_app() -> FastAPI:
    """Create the app instance and attach all production routes."""
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="OWLBAN Production AI Platform",
        version="1.1.0",
        description="Production-ready application layer for OWLBAN AI, quantum, and finance workflows.",
        debug=settings.debug,
    )

    app.state.settings = settings
    app.state.ai_service = AIService()
    app.state.store = AppStore(settings.database_path)

    app.add_middleware(
        RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> Dict[str, Any]:
        payload = app.state.ai_service.health()
        return HealthResponse(
            status=payload["status"],
            ai_available=payload["ai_available"],
            quantum_available=payload["quantum_available"],
            environment=payload["environment"],
        )

    @app.get("/status")
    async def status() -> Dict[str, Any]:
        return {
            "status": "ok",
            "environment": settings.environment,
            "events_logged": app.state.store.count_events(),
        }

    @app.post("/ai/optimize", response_model=OptimizationResponse)
    async def optimize(request: OptimizationRequest, username: str = Depends(verify_credentials)) -> OptimizationResponse:
        result = app.state.ai_service.optimize_assets(
            [asset.model_dump() for asset in request.assets],
            iterations=request.iterations,
            target_return=request.target_return,
        )
        app.state.store.log_event(
            "ai_optimize",
            {"username": username, "asset_count": len(request.assets)},
        )
        return OptimizationResponse(
            status=result["status"],
            best_asset=result["best_asset"],
            best_score=float(result["best_score"]),
            mean_score=float(result["mean_score"]),
            winner_index=int(result["winner_index"]),
            metrics=result["metrics"],
        )

    @app.post("/ai/risk")
    async def evaluate_risk(request: PortfolioRiskRequest, username: str = Depends(verify_credentials)) -> Dict[str, Any]:
        portfolio = [asset.model_dump() for asset in request.portfolio]
        response = app.state.ai_service.evaluate_risk(portfolio, request.stress_scenario)
        app.state.store.log_event(
            "ai_risk",
            {
                "username": username,
                "asset_count": len(request.portfolio),
                "scenario": request.stress_scenario,
            },
        )
        return response

    @app.post("/quantum/demo")
    async def quantum_demo(username: str = Depends(verify_credentials)) -> Dict[str, Any]:
        response = app.state.ai_service.quantum_demo()
        app.state.store.log_event("quantum_demo", {"username": username})
        return response

    return app


app = create_app()
