# Run the API and dashboard locally

## Prerequisites

- Python 3.10+ installed
- Docker (optional)

## Set up the workspace environment

```powershell
./scripts/setup_env.ps1
```

## Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## API authentication (HTTP Basic auth)

The API is protected by HTTP Basic auth. `API_PASSWORD` is required by
**`api_server.py`** for direct `uvicorn` invocations. The two supported ways
to provide it:

1. **Recommended** — copy `.env.example` to `.env`, set `API_PASSWORD` (and any
   other vars you want). `main.py` loads `.env` automatically:
   ```powershell
   copy .env.example .env
   # edit .env -> set API_PASSWORD=...
   ```
2. **Zero-config (local dev only)** — `python main.py` generates a strong
   random `API_PASSWORD` if one is not set and prints it to the console, so
   the command below works out of the box. Running
   `uvicorn api_server:fastapi_app` directly (without `main.py`) still
   requires `API_PASSWORD` to be set in the environment.

> `docker` and `stripe` Python SDKs are **optional**. Without them the
> platform runs in simulation/mock mode ("using mock mode",
> "dummy Stripe API key") and stays fully functional. Install them only if
> you need real container deployments or live Stripe payments:
> `pip install docker stripe`.

## Start the API (background or separate terminal)

```powershell
.\.venv\Scripts\python.exe main.py
# or
.\.venv\Scripts\python.exe -m uvicorn api_server:fastapi_app --host 0.0.0.0 --port 8000
```

## Start the Streamlit dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Docker (build dashboard image)

```powershell
docker build -f Dockerfile.dashboard -t owlban-dashboard .
docker run -p 8501:8501 owlban-dashboard
```

If you want, I can attempt to run these commands here and capture logs; tell me to proceed.
