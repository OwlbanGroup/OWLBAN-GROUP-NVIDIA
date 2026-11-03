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

## OSCAR BROOME Revenue service (`OSCAR-BROOME-REVENUE/`)

Node/Express + JWT service on port 3001 (browser UI at `/`, dashboard at
`/dashboard`). It seeds Oscar's owner account on first boot.

> `OSCAR-BROOME-REVENUE` is recorded as a **gitlink**, so edits made inside it
> are tracked by neither repo. The update to that service is captured in
> [`.tools/apply_update.py`](.tools/apply_update.py) — idempotent, supports
> `--check` and `--revert`, and stops on drift instead of guessing:
>
> ```powershell
> .\.venv\Scripts\python.exe .tools\apply_update.py --check   # what would change
> .\.venv\Scripts\python.exe .tools\apply_update.py           # apply
> ```

```powershell
cd OSCAR-BROOME-REVENUE
npm install
$env:OSCAR_OWNER_PASSWORD = 'a-strong-password'   # optional, see below
npm start
```

- `OSCAR_OWNER_EMAIL` / `OSCAR_OWNER_USERNAME` — owner identity
  (defaults `oscar.broome@jpmorgan.com` / `Oscar Broome`).
- `OSCAR_OWNER_PASSWORD` — pins the owner password. **No password is baked
  into the repo**: when unset, the first boot generates a random one and prints
  it to the console once. Set it in `.env`/compose for reproducible creds.
- `OSCAR_JWT_SECRET` — JWT signing key (random per process when unset).
- `OSCAR_DATA_DIR` — where `users.json` / `tokens.json` live (tests use a temp dir).

End-to-end test (headless browser logs in as Oscar and checks the dashboard):

```powershell
cd OSCAR-BROOME-REVENUE
$env:OSCAR_OWNER_PASSWORD = 'a-strong-password'   # required to run the e2e
npm test
```

### Upstream clone check (`verify-oscar-login.mjs`)

`OSCAR-BROOME-REVENUE/verify-oscar-login.mjs` is a copy of the upstream
[OwlbanGroup/OSCAR-BROOME-REVENUE](https://github.com/OwlbanGroup/OSCAR-BROOME-REVENUE)
verifier; it belongs in that clone (here: `C:\Users\bsean\oscar-revenue-work`)
and needs the clone's ESM server, so run it from the clone root. That clone's
`middleware/authOverride.js` refuses to start unless all three override
passwords are set:

```powershell
cd C:\Users\bsean\oscar-revenue-work
$env:OVERRIDE_ADMIN_PASS='...'; $env:OVERRIDE_MANAGER_PASS='...'; $env:OVERRIDE_SUPER_ADMIN_PASS='...'
node scripts/verify-oscar-login.mjs
```

`OSCAR_PASSWORD` additionally runs the authenticated login checks. The verifier
no longer requires a `data/` directory to exist in the clone.

## Known slow / hanging tests

`pytest tests` runs to completion but takes several minutes on CPU-only hosts:
`torch` import alone is ~15 s here, and `tests/test_quantum_ai.py::
TestQuantumMarketPredictor::test_market_predictor` plus the QSVC tests in
`tests/test_quantum_machine_learning.py` perform real quantum-kernel training
(exact statevector simulation) — minutes of CPU each. For a fast local run:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests -q `
  --ignore=tests/test_quantum_ai.py --ignore=tests/test_quantum_machine_learning.py
```
