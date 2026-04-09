#!/usr/bin/env python3
"""
Docker Engine API Example Client
Matches https://docs.docker.com/reference/api/engine/

Requires: pip install docker
Usage: python docker/engine-api-example.py
"""

import docker
from typing import List, Dict

client = docker.from_env()

def list_containers() -> List[Dict]:
    """List all containers (Engine API /containers/json)"""
    containers = client.containers.list(all=True)
    return [{"id": c.id[:12], "name": c.name, "status": c.status, "image": c.image.tags[0] if c.image.tags else c.image.id} for c in containers]

def start_gateway() -> str:
    """Start gateway container example"""
    try:
        container = client.containers.get("jpmorgan_gateway")
        container.start()
        return f"Started {container.name}"
    except docker.errors.NotFound:
        return "Gateway container not found - run 'make up' first"

def create_health_container() -> str:
    """Create example healthcheck container (Engine API /containers/create)"""
    container = client.containers.create(
        "alpine:3.18",
        command="sleep 3600",
        name="docker-ref-health",
        healthcheck={"test": ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8000/health"], "interval": 30, "timeout": 10, "retries": 3},
        network="jpm-net"
    )
    container.start()
    return f"Created health container: {container.id[:12]}"

if __name__ == "__main__":
    print("Docker Engine API Examples:")
    print("1. List containers:", list_containers())
    print("2. Start gateway:", start_gateway())
    print("3. Create health container:", create_health_container())
    print("\nMatches Docker Engine API v1.45+: https://docs.docker.com/reference/api/engine/")

