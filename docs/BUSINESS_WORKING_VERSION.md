# Business Working Version

Release status: local production-ready

Date: 2026-11-18

## Included

- FastAPI application service layer with health, optimization, risk, and quantum demo endpoints.
- Basic, bearer-JWT, and API-key authentication.
- SQLite event telemetry with portfolio payloads excluded from audit records.
- Structured request logging and request-ID propagation.
- Process-local rate limiting with burst and refill behavior.
- Production Docker image running as a non-root user.
- TLS-terminating NGINX Compose stack.
- `.env.production` setup with generated random secrets.
- Local development TLS certificate and private key.
- Root `requirements.txt` compatibility entrypoint.
- Cross-platform lint script for Windows and Unix environments.

## Verification

The following checks have passed:

- Full Python pytest suite.
- Ruff lint.
- Pylint at 10.00/10.
- Docker Compose configuration validation.
- Token-bucket refill test.
- Certificate and private-key key-pair validation.

Generated secrets and certificate files are excluded from Git by `.gitignore`.

## Start Locally

The setup script generates `.env.production` and a local self-signed certificate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-production.ps1
docker compose -f docker-compose.production.yml config
docker compose -f docker-compose.production.yml up --build -d
```

For a fresh local setup that replaces existing generated files:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-production.ps1 -Force -Domain localhost
```

Check service status and health:

```powershell
docker compose -f docker-compose.production.yml ps
curl.exe -k https://localhost/health
```

Stop the local stack:

```powershell
docker compose -f docker-compose.production.yml down
```

## Public Release Requirements

The self-signed certificate must not be used for public deployment. Before exposing the service to users:

1. Replace `deploy/tls/fullchain.pem` and `deploy/tls/privkey.pem` with a trusted certificate for the public hostname.
2. Replace the example CORS origin with the exact browser origin.
3. Store `.env.production` secrets in an approved secret manager or secure deployment environment.
4. Rotate any generated local credential before production use.
5. Replace process-local rate limiting with a shared backend for multiple replicas.
6. Replace SQLite telemetry with a managed database for multi-instance deployments.
7. Configure backups, monitoring, alerting, certificate renewal, and incident recovery.
8. Verify authentication, TLS, persistence, restart behavior, and rollback in the deployment environment.

## Operational Notes

- Never commit `.env.production`, `fullchain.pem`, or `privkey.pem`.
- Do not expose application port 8000 publicly; use the NGINX proxy.
- Keep the in-memory rate limiter and SQLite store within the documented single-instance limitations.
- Treat financial and quantum output in the app as application capabilities requiring validated business integrations before use for regulated decisions.
- Use [PUBLIC_TLS_DNS.md](PUBLIC_TLS_DNS.md) for the GoDaddy DNS and trusted-certificate process.
