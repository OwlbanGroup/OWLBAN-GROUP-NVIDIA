# Production-ready application layer

This repository now includes a cleaner application layer under `app/` that is intended for service-oriented deployment and local development.

## Overview

The production layer provides:

- a FastAPI app factory for consistent startup
- typed request/response models
- a small service layer wrapping AI and quantum workflows
- health, optimization, and risk endpoints
- end-to-end sample scripts for local execution

## Authentication

The app layer supports three auth modes for production-friendly access:

- HTTP Basic Auth
- Bearer JWT tokens via `Authorization: Bearer <token>`
- API keys via `X-API-Key: <key>`

Use the same credentials configured in `API_USERNAME` / `API_PASSWORD` for basic auth, or set `JWT_SECRET` for signed bearer tokens.

## Run locally

```bash
python main.py
```

or directly:

```bash
python -m app.main
```

The app listens on `127.0.0.1:8000` by default and respects `API_HOST`, `API_PORT`, and `DEBUG` environment variables.

## Production deployment

The production Docker Compose stack keeps the API off the public host ports and terminates TLS at NGINX. The app container runs as a non-root user, read-only root filesystem, a persistent SQLite data volume, and one Uvicorn worker (the rate limiter is process-local).

1. Copy `.env.app.example` to `.env.production`.
2. Set `API_PASSWORD` and `JWT_SECRET` to unique secrets. `JWT_SECRET` must be at least 32 bytes. Never commit `.env.production`.
3. If using API keys, configure `APP_API_KEYS` as a JSON object mapping each secret key to a principal, for example `{"replace-with-a-random-key":"service-client"}`. Keep the keys in a secret manager or protected environment file.
4. Install your TLS certificate and private key at `deploy/tls/fullchain.pem` and `deploy/tls/privkey.pem`; restrict private-key file access and do not commit either file.
5. Update `CORS_ORIGINS` to the exact browser origin(s) that use the API.
6. Start with `docker compose -f docker-compose.production.yml up --build -d`.
7. Check `docker compose -f docker-compose.production.yml ps` and query `https://your-host/health`.

The app validates required production secrets at startup. Logs are JSON lines on stdout, and application responses include an `X-Request-ID`. Do not expose port 8000 directly. For multiple replicas, use a shared rate-limit backend instead of the in-memory limiter and migrate SQLite telemetry to a managed database.

## Endpoints

- `GET /health` — checks application health
- `GET /status` — reports runtime configuration
- `POST /ai/optimize` — evaluates an asset optimization request
- `POST /ai/risk` — calculates risk for a portfolio
- `POST /quantum/demo` — prepares a Bell-pair quantum state

## Example usage

```bash
python examples/quantum_ai_workflow.py
```

This runs a local workflow that checks service health, optimizes a portfolio, evaluates stress risk, and demonstrates the quantum package.

You can also call the API directly from Python:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ai/optimize",
    json={
        "assets": [
            {"symbol": "NVDA", "value": 150000.0, "volatility": 0.18, "score": 0.92},
            {"symbol": "MSFT", "value": 130000.0, "volatility": 0.14, "score": 0.88},
        ],
        "iterations": 3,
        "target_return": 0.1,
    },
)
print(response.json())
```

## Production guidance

- keep environment variables in `.env` or a secret manager
- use managed credentials instead of hard-coded passwords
- configure TLS at the reverse proxy or managed ingress
- set exact allowed browser origins; wildcard CORS is rejected
- use structured logs and propagate request IDs into centralized observability
- expand the service layer instead of creating custom logic directly in route handlers
