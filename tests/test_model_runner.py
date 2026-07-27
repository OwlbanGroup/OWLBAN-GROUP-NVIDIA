import pytest
import httpx
import subprocess
import time
import docker

@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()

@pytest.fixture(scope="session")
def model_runner_container(docker_client):
    """Spin up model-runner for tests."""
    subprocess.run(["make", "model-up"], check=True)
    time.sleep(30)  # Wait for Ollama pull/start
    container = docker_client.containers.get("jpmorgan_model_runner")
    yield container
    subprocess.run(["make", "model-down"], check=True)

def test_model_runner_health(model_runner_container):
    """Test model-runner health and Ollama API."""
    resp = httpx.get("http://localhost:11434/api/tags")
    assert resp.status_code == 200
    tags = resp.json()
    assert "llama3.1" in str(tags)  # Default model loaded

def test_model_runner_chat(model_runner_container):
    """Test simple chat completion."""
    payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": False
    }
    resp = httpx.post("http://localhost:11434/api/chat", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert "message" in result
    assert "4" in result["message"]["content"]

def test_model_runner_generate(model_runner_container):
    """Test generate endpoint used by MCP."""
    payload = {
        "model": "llama3.1",
        "prompt": "Analyze risk: balance $10000.",
        "stream": False
    }
    resp = httpx.post("http://localhost:11434/api/generate", json=payload)
    assert resp.status_code == 200
    assert "response" in resp.json()
    assert len(resp.json()["response"]) > 10

def test_mcp_llm_tool(model_runner_container):
    """Test MCP analyze_financial_data tool (requires mcp-server up)."""
    subprocess.run(["make", "mcp-up"], check=True)
    time.sleep(10)
    payload = {
        "tool": "analyze_financial_data",
        "params": {"financial_data": '{"balance": 10000, "transactions": []}'}
    }
    resp = httpx.post("http://localhost:8080/mcp/call", json=payload)
    assert resp.status_code == 200
    content = resp.text.strip()
    assert "risk" in content.lower() or "analysis" in content.lower()
    subprocess.run(["make", "mcp-down"], check=True)

@pytest.mark.slow
def test_gpu_usage(model_runner_container, request):
    """Test GPU in model-runner (slow)."""
    if not request.config.getoption("--runslow"):
        pytest.skip("use pytest --runslow")
    logs = model_runner_container.logs().decode()
    assert "CUDA" in logs or "GPU" in logs or "nvidia" in logs
