"""
Local entrypoint to start the OWLBAN API server.

Run this with `python main.py` to start the FastAPI app via Uvicorn.
"""

import uvicorn


def main():
	uvicorn.run("api_server:fastapi_app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
	main()
