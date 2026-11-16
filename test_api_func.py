"""Test API server functionality on CPU."""

import asyncio
import os

import pytest


@pytest.mark.parametrize("password", ["test"])
def test_api_server_smoke(password):
    os.environ["API_PASSWORD"] = password

    import api_server
    from combined_nim_owlban_ai import CombinedSystem

    result = asyncio.run(api_server.get_system_status())
    assert result.timestamp
    assert isinstance(result.services, dict)
    assert "combined_system" in result.services

    cs = CombinedSystem()
    cs.initialize()
    prediction = cs.run_inference({"test": "data"})
    assert isinstance(prediction, dict)
    assert "prediction" in prediction


def main():
    test_api_server_smoke()


if __name__ == "__main__":
    main()
