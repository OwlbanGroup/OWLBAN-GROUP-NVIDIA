#!/usr/bin/env python3
"""
Shared test server utilities for phase endpoint scripts.
Starts/stops an in-process Flask app with the PFM blueprint for reliable local testing.
"""

import os
import sys
import threading
import time
from typing import Optional

import requests
from flask import Flask
from flask_cors import CORS
from werkzeug.serving import make_server


class _ServerThread(threading.Thread):
    def __init__(self, app: Flask, host: str = "127.0.0.1", port: int = 5000):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self._server = make_server(host, port, app)
        self._ctx = app.app_context()
        self._ctx.push()

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()
        self._ctx.pop()


_server_thread: Optional[_ServerThread] = None


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _ensure_sys_path():
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def create_test_app() -> Flask:
    _ensure_sys_path()
    os.environ["TESTING"] = "1"

    from blueprints.pfm import pfm_bp

    app = Flask(__name__)
    app.config["TESTING"] = True
    CORS(app)
    app.register_blueprint(pfm_bp, url_prefix="/pfm")
    return app


def is_server_healthy(base_url: str = "http://127.0.0.1:5000") -> bool:
    # Use an existing route that should be available when PFM is mounted
    try:
        resp = requests.get(f"{base_url}/pfm/accounts", timeout=1.5)
        return resp.status_code in (200, 400)
    except requests.RequestException:
        return False


def ensure_local_test_server(base_url: str = "http://127.0.0.1:5000"):
    """
    Start in-process local server if the expected PFM app is not already healthy.
    Returns tuple: (started_here: bool, server_thread_or_none)
    """
    global _server_thread

    if is_server_healthy(base_url):
        return False, None

    app = create_test_app()
    _server_thread = _ServerThread(app, host="127.0.0.1", port=5000)
    _server_thread.start()

    # Wait for readiness
    for _ in range(20):
        if is_server_healthy(base_url):
            return True, _server_thread
        time.sleep(0.25)

    raise RuntimeError("Local test server failed to start or become healthy")


def stop_local_test_server():
    global _server_thread
    if _server_thread is not None:
        _server_thread.shutdown()
        _server_thread = None
