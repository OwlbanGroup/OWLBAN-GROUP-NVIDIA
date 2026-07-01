import asyncio

import api_server


def test_get_system_status_returns_response_without_crashing():
    result = asyncio.run(api_server.get_system_status())

    assert result.timestamp
    assert isinstance(result.services, dict)
    assert isinstance(result.monitoring, dict)
