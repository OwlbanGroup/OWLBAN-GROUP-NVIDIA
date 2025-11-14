"""
Module for integrating GitHub MCP Server with JPMorgan Financial APIs

This module initializes the MCP Server client and provides functions
to interact with the MCP Server for repository management, issue handling,
and other GitHub platform capabilities.

The MCP Server is run as a subprocess or Docker container as configured.
Supports both real MCP server and mock implementation for testing.
"""

import subprocess
import json
import shlex
import sys
import os
import logging
from typing import Optional, Dict, Any, List

# Add parent directory to sys.path for absolute import
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import config

logger = logging.getLogger(__name__)

class MockMCPServerClient:
    """Mock implementation for testing when MCP server is not available"""

    def __init__(self):
        logger.info("Using mock MCP client for testing")

    def list_repositories(self, query: str = "", per_page: int = 10) -> List[Dict[str, Any]]:
        """Return mock repository data"""
        mock_repos = [
            {"name": "flask", "full_name": "pallets/flask", "description": "A simple framework for building complex web applications."},
            {"name": "requests", "full_name": "psf/requests", "description": "Python HTTP for Humans."},
            {"name": "pandas", "full_name": "pandas-dev/pandas", "description": "Flexible and powerful data analysis / manipulation library for Python"},
        ]
        # Filter by query if provided
        if query:
            mock_repos = [repo for repo in mock_repos if query.lower() in repo["name"].lower()]
        return mock_repos[:per_page]

    def list_issues(self, owner: str, repo: str, state: str = "open", per_page: int = 10) -> List[Dict[str, Any]]:
        """Return mock issue data"""
        mock_issues = [
            {"title": f"Mock issue 1 for {owner}/{repo}", "state": state, "number": 1},
            {"title": f"Mock issue 2 for {owner}/{repo}", "state": state, "number": 2},
            {"title": f"Mock issue 3 for {owner}/{repo}", "state": state, "number": 3},
        ]
        return mock_issues[:per_page]

    def create_issue(self, owner: str, repo: str, title: str, body: str = "", assignees: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return mock issue creation response"""
        return {
            "title": title,
            "body": body,
            "number": 999,
            "html_url": f"https://github.com/{owner}/{repo}/issues/999"
        }

class MCPServerClient:
    def __init__(self):
        self.command = config.MCP_SERVER_COMMAND
        self.token = config.GITHUB_PERSONAL_ACCESS_TOKEN
        self.toolsets = config.MCP_SERVER_TOOLSETS
        self.host = config.MCP_SERVER_HOST
        self.use_mock = not self.token or not self._is_server_available()
        if self.use_mock:
            logger.warning("MCP server not available, using mock implementation")
            self.mock_client = MockMCPServerClient()
        else:
            logger.info("MCP server available, using real implementation")

    def _is_server_available(self) -> bool:
        """Check if MCP server is available"""
        try:
            # Try to run a simple command to check availability
            cmd_list = shlex.split(self.command + " --help")
            process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.token} if self.token else None
            )
            stdout, stderr = process.communicate(timeout=10)
            return process.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def _run_command(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the MCP Server command with optional JSON input and return JSON output.
        """
        try:
            cmd_list = shlex.split(self.command)
            env = None
            if self.token:
                env = {"GITHUB_PERSONAL_ACCESS_TOKEN": self.token}
            process = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            input_str = json.dumps(input_data) if input_data else None
            stdout, stderr = process.communicate(input=input_str.encode() if input_str else None)
            if process.returncode != 0:
                raise RuntimeError(f"MCP Server command failed: {stderr.decode()}")
            return json.loads(stdout.decode())
        except Exception as e:
            raise RuntimeError(f"Error running MCP Server command: {str(e)}")

    def list_repositories(self, query: str = "", per_page: int = 10) -> List[Dict[str, Any]]:
        """
        List repositories matching the query.
        """
        if self.use_mock:
            return self.mock_client.list_repositories(query, per_page)

        try:
            input_data = {
                "tool": "repos",
                "action": "search_repositories",
                "parameters": {
                    "query": query,
                    "perPage": per_page
                }
            }
            result = self._run_command(input_data)
            return result.get("repositories", [])
        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            # Fallback to mock data
            return self.mock_client.list_repositories(query, per_page)

    def list_issues(self, owner: str, repo: str, state: str = "open", per_page: int = 10) -> List[Dict[str, Any]]:
        """
        List issues for a repository.
        """
        if self.use_mock:
            return self.mock_client.list_issues(owner, repo, state, per_page)

        try:
            input_data = {
                "tool": "issues",
                "action": "list_issues",
                "parameters": {
                    "owner": owner,
                    "repo": repo,
                    "state": state,
                    "perPage": per_page
                }
            }
            result = self._run_command(input_data)
            return result.get("issues", [])
        except Exception as e:
            logger.error(f"Failed to list issues: {e}")
            # Fallback to mock data
            return self.mock_client.list_issues(owner, repo, state, per_page)

    def create_issue(self, owner: str, repo: str, title: str, body: str = "", assignees: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new issue in a repository.
        """
        if self.use_mock:
            return self.mock_client.create_issue(owner, repo, title, body, assignees)

        try:
            input_data = {
                "tool": "issues",
                "action": "create_issue",
                "parameters": {
                    "owner": owner,
                    "repo": repo,
                    "title": title,
                    "body": body,
                    "assignees": assignees or []
                }
            }
            result = self._run_command(input_data)
            return result
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            # Fallback to mock data
            return self.mock_client.create_issue(owner, repo, title, body, assignees)

# Singleton MCP Server client instance
mcp_client = MCPServerClient()
