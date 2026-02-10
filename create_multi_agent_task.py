#!/usr/bin/env python3
"""
Script to create a Multi-Agent Task for Blackbox AI integration
"""

import os
import requests
import json
from typing import Dict, Any

def create_multi_agent_task(api_key: str, repo_url: str, prompt: str, selected_agents: list, branch: str = "main") -> Dict[str, Any]:
    """
    Create a multi-agent task using Blackbox AI API

    Args:
        api_key: Blackbox API key (starts with 'bb_')
        repo_url: GitHub repository URL
        prompt: Task description
        selected_agents: List of agent configurations
        branch: Branch to work on (default: main)

    Returns:
        API response as dictionary
    """
    url = "https://cloud.blackbox.ai/api/tasks"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "repoUrl": repo_url,
        "selectedBranch": branch,
        "selectedAgents": selected_agents
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating multi-agent task: {e}")
        return {"error": str(e)}

def main():
    # Get API key from environment or prompt
    api_key = os.getenv('BLACKBOX_API_KEY')
    if not api_key:
        api_key = input("Enter your Blackbox API key (starts with 'bb_'): ").strip()
        if not api_key.startswith('bb_'):
            print("Invalid API key format. Should start with 'bb_'")
            return

    # Repository details
    repo_url = "https://github.com/ESADavid/jpmorgan_financial_apis.git"
    branch = "main"

    # Task prompt
    prompt = """
    Integrate Blackbox AI into the JPMorgan Financial APIs project. Complete the following tasks:

    1. Add Blackbox AI configuration settings to config.py
    2. Update AI service initialization in src/ai_service.py to support Blackbox AI
    3. Test Blackbox AI integration with sample queries
    4. Update requirements.txt if needed for new dependencies

    Ensure the integration follows the existing code patterns and maintains backward compatibility with OpenAI.
    """

    # Selected agents for multi-agent task
    selected_agents = [
        {
            "agent": "claude",
            "model": "blackboxai/anthropic/claude-sonnet-4.5"
        },
        {
            "agent": "blackbox",
            "model": "blackboxai/blackbox-pro"
        },
        {
            "agent": "gemini",
            "model": "gemini-2.0-flash-exp"
        },
        {
            "agent": "codex",
            "model": "gpt-5.2-codex"
        }
    ]

    print("Creating Multi-Agent Task for Blackbox AI integration...")
    print(f"Repository: {repo_url}")
    print(f"Branch: {branch}")
    print(f"Agents: {len(selected_agents)}")
    print(f"Prompt: {prompt.strip()[:100]}...")

    # Create the task
    result = create_multi_agent_task(api_key, repo_url, prompt, selected_agents, branch)

    if "error" in result:
        print(f"❌ Failed to create task: {result['error']}")
        return

    task_id = result.get("task", {}).get("id")
    task_url = result.get("taskUrl")

    print("✅ Multi-Agent Task created successfully!")
    print(f"Task ID: {task_id}")
    print(f"Task URL: {task_url}")
    print("\n📋 Task Details:")
    print(f"- Prompt: {prompt.strip()}")
    print(f"- Repository: {repo_url}")
    print(f"- Branch: {branch}")
    print(f"- Agents: {[agent['agent'] for agent in selected_agents]}")

    print("\n🔄 Task Status: Pending")
    print("You can monitor progress at the task URL above.")
    print("The agents will start working on the integration shortly.")

if __name__ == "__main__":
    main()
