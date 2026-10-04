"""
Local entrypoint to start the OWLBAN API server.

Run this with `python main.py` to start the FastAPI app via Uvicorn.

The API is protected by HTTP Basic auth (API_USERNAME / API_PASSWORD). If
API_PASSWORD is not supplied via the environment, a strong random password is
generated for this run and printed to the console so `python main.py` works
out of the box. For production / repeatable deployments, set API_PASSWORD (and
the other variables documented in .env.example) explicitly.
"""

import os
import secrets

import uvicorn


def _ensure_api_password() -> None:
    """Guarantee an API_PASSWORD exists so the server can start.

    Keeps the auth guard inside ``api_server`` intact: HTTP Basic auth is never
    disabled, a credential is always present. We only fill in a random default
    when the operator has not provided one, which makes the documented
    `python main.py` entry point work for local development.
    """
    if os.getenv("API_PASSWORD"):
        return
    generated = secrets.token_urlsafe(18)
    os.environ["API_PASSWORD"] = generated
    api_user = os.getenv("API_USERNAME", "owlban_admin")
    print("\n" + "=" * 64)
    print("API_PASSWORD was not set; generated a random one for this run.")
    print("  API_USERNAME:", api_user)
    print("  API_PASSWORD:", generated)
    print("Hit the API with:  curl -u "
          + api_user + ":" + generated + " http://127.0.0.1:8000/status")
    print("For production, set API_PASSWORD in your environment "
          "(see .env.example).")
    print("=" * 64 + "\n")


def _load_dotenv() -> None:
    """Minimal .env loader (no third-party dependency).

    Reads an optional ``.env`` file in the current directory and exports any
    KEY=VALUE lines into ``os.environ`` that are not already set.
    """
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except OSError:
        # A missing/unreadable .env is non-fatal; fall back to defaults.
        pass


def main():
    _load_dotenv()
    _ensure_api_password()
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api_server:fastapi_app", host=api_host,
                port=api_port, log_level="info")


if __name__ == "__main__":
    main()

