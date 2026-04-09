import pytest
import httpx
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_mcp_server_tools():
    async with AsyncClient(base_url="http://localhost:8080") as client:
        resp = await client.get("/mcp/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert "get_accounts" in str(tools)
        assert "get_balance" in str(tools)
        assert len(tools) > 0

@pytest.mark.asyncio
async def test_mcp_health():
    async with AsyncClient(base_url="http://localhost:8080") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

@pytest.mark.parametrize("tool", ["get_accounts", "get_balance", "transfer_funds", "get_transactions"])
@pytest.mark.asyncio
async def test_mcp_call(tool):
    # Mock call - assumes gateway mock data
    async with AsyncClient(base_url="http://localhost:8080") as client:
        resp = await client.post("/mcp/call", json={
            "tool": tool,
            "params": {"account_id": "test123"}
        })
        assert resp.status_code in [200, 201]

