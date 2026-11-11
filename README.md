# OWLBAN-GROUP-NVIDIA

[![CI](https://github.com/OwlbanGroup/OWLBAN-GROUP-NVIDIA/actions/workflows/ci.yml/badge.svg)](https://github.com/OwlbanGroup/OWLBAN-GROUP-NVIDIA/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: ISC](https://img.shields.io/badge/license-ISC-green.svg)](LICENSE)

**OWLBAN GROUP & NVIDIA AI** — Enterprise-grade quantum AI platform with NVIDIA NIM integration, FastAPI backend, Streamlit dashboard, comprehensive authentication, and multi-database support.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/OwlbanGroup/OWLBAN-GROUP-NVIDIA.git
cd OWLBAN-GROUP-NVIDIA

# Install Python dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration (API_PASSWORD, API_USERNAME, etc.)

# Start the API server
python main.py

# In another terminal, start the Streamlit dashboard
streamlit run web_dashboard.py
```

The API server generates a random `API_PASSWORD` on first run if none is set — check the console output.

## 📋 Features

| Category | Capabilities |
|----------|-------------|
| **AI / Quantum** | NVIDIA NIM integration, quantum machine learning, quantum financial models, RAPIDS, Triton inference, DCGM GPU monitoring |
| **Authentication** | JWT access/refresh tokens, bcrypt password hashing, TOTP/MFA, OAuth2, API keys, rate limiting, CSRF protection |
| **Database** | SQLite (default), PostgreSQL, MongoDB, Redis with graceful fallbacks |
| **Banking** | JPMorgan API integration — payments, treasury, risk analysis, market data |
| **Dashboard** | Streamlit web UI for monitoring AI systems, GPU status, revenue optimization, quantum predictions |
| **Security** | Security headers (CSP, HSTS, X-Frame-Options), CORS hardening, input validation, audit logging |

## 📁 Project Structure

```
OWLBAN-GROUP-NVIDIA/
├── api_server.py          # FastAPI REST API server
├── main.py                # Application entrypoint (loads .env, starts uvicorn)
├── web_dashboard.py       # Streamlit dashboard
├── auth_lib.py            # Authentication library (JWT, bcrypt, TOTP, OAuth2)
├── database_manager.py    # Multi-database manager (SQLite/PG/Mongo/Redis)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── Dockerfile.api         # API container
├── Dockerfile.dashboard   # Dashboard container
├── Dockerfile.qiskit      # Quantum container
├── docker-compose.yml     # Multi-service orchestration
├── combined_nim_owlban_ai/# NVIDIA NIM + OWLBAN AI integration package
├── new_products/          # Revenue optimizer, anomaly detection, telehealth, etc.
├── performance_optimization/# RL agent, advanced anomaly detection
├── quantum_financial_ai/  # Quantum ML pipelines, portfolio optimization
├── middleware/            # Rate limiting, security headers, CSRF
├── monitoring/            # Prometheus, Grafana, alert rules
├── tests/                 # Test suite (pytest)
└── scripts/               # Deployment and setup scripts
```

## 🔐 Authentication

The API is protected by HTTP Basic auth. Set `API_USERNAME` and `API_PASSWORD` in your environment (or `.env` file). If `API_PASSWORD` is unset, `main.py` generates a random one and prints it to the console.

The `auth_lib.py` module additionally provides:
- JWT access + refresh tokens
- Password reset flow
- TOTP-based MFA (Google Authenticator compatible)
- API key generation and verification
- Account lockout after failed login attempts

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-timeout

# Run all tests
pytest tests/ -q

# Run a specific test file
pytest tests/test_auth_system.py -q
```

## 🐳 Docker

```bash
# Build and start all services
docker compose up -d

# API server
docker compose up api

# Dashboard
docker compose up dashboard
```

## 📖 Documentation

- `.env.example` — Full environment variable reference
- `RUN.md` — Operational runbook
- `PRODUCTION_DEPLOYMENT_PLAN.md` — Production deployment guide
- `USER_AUTHENTICATION_GUIDE.md` — Authentication system documentation
- `COMPLETE_MASTER_PLAN.md` — Strategic master plan

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes and add tests
4. Run the test suite (`pytest tests/ -q`)
5. Commit and push (`git push origin feature/improvement`)
6. Open a pull request

## 📄 License

ISC License — see [LICENSE](LICENSE) for details.

## 🏢 About

OWLBAN GROUP — Advanced quantum AI and enterprise financial technology, powered by NVIDIA.
