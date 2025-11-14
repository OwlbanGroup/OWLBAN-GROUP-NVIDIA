#!/usr/bin/env python3
"""
Test script for MCP Server integration
"""

import sys
import os

# Set environment variables for testing
os.environ['ALLOW_MISSING_TOKENS'] = 'true'
os.environ['SECRET_KEY'] = 'test-secret-key-for-testing'

sys.path.append(os.path.dirname(__file__))

from src.mcp_integration import mcp_client

def test_mcp_client():
    """Test the MCP client functionality"""
    print("Testing MCP Client...")

    try:
        # Test list_repositories
        print("Testing list_repositories...")
        repos = mcp_client.list_repositories("flask", 2)
        print(f"Found {len(repos)} repositories")
        for repo in repos:
            print(f"- {repo.get('name', 'Unknown')}")

        # Test list_issues
        print("\nTesting list_issues...")
        issues = mcp_client.list_issues("microsoft", "vscode", "open", 2)
        print(f"Found {len(issues)} issues")
        for issue in issues:
            print(f"- {issue.get('title', 'Unknown')}")

        # Test create_issue (commented out to avoid creating real issues)
        # print("\nTesting create_issue...")
        # result = mcp_client.create_issue("test", "test", "Test Issue", "Test body")
        # print(f"Created issue: {result}")

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

    return True

if __name__ == "__main__":
    success = test_mcp_client()
    sys.exit(0 if success else 1)
