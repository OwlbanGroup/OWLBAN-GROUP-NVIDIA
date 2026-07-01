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
