Run the API and dashboard locally

Prerequisites
- Python 3.11+ installed
- Docker (optional)

Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Start the API (background or separate terminal)

```powershell
python main.py
# or
python -m uvicorn api_server:fastapi_app --host 0.0.0.0 --port 8000
```

Start the Streamlit dashboard

```powershell
python -m streamlit run web_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Docker (build dashboard image)

```powershell
docker build -f Dockerfile.dashboard -t owlban-dashboard .
docker run -p 8501:8501 owlban-dashboard
```

If you want, I can attempt to run these commands here and capture logs; tell me to proceed.