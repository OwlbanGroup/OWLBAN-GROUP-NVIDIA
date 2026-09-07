import asyncio
import os

import pytest

if not os.environ.get("API_PASSWORD"):
    pytest.skip("api_server requires API_PASSWORD environment variable", allow_module_level=True)

import api_server


def test_get_system_status_returns_response_without_crashing():
    result = asyncio.run(api_server.get_system_status())

    assert result.timestamp
    assert isinstance(result.services, dict)
    assert isinstance(result.monitoring, dict)
