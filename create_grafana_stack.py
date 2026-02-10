import os
import requests
import json
import time

# Grafana Cloud API endpoints
GRAFANA_CLOUD_API_BASE = "https://grafana.com/api"
STACK_NAME = "jpmorgan-financial-stack"
STACK_REGION = "us-central1"  # Default region, can be changed

def create_grafana_stack(api_key, name=STACK_NAME, region=STACK_REGION):
    """Create a new Grafana Cloud stack."""
    url = f"{GRAFANA_CLOUD_API_BASE}/stacks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "name": name,
        "region": region
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        stack_data = response.json()
        print(f"Stack '{name}' created successfully!")
        print(f"Stack URL: {stack_data['url']}")
        print(f"Stack Slug: {stack_data['slug']}")
        return stack_data
    else:
        print(f"Failed to create stack: {response.status_code} - {response.text}")
        return None

def get_stack_details(api_key, stack_slug):
    """Get details of the created stack."""
    url = f"{GRAFANA_CLOUD_API_BASE}/stacks/{stack_slug}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get stack details: {response.status_code} - {response.text}")
        return None

def import_dashboard(stack_url, api_key, dashboard_json):
    """Import the dashboard to the stack."""
    url = f"{stack_url}/api/dashboards/db"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Load the dashboard JSON
    with open('grafana_dashboard.json', 'r') as f:
        dashboard = json.load(f)

    # Prepare the payload
    payload = {
        "dashboard": dashboard['dashboard'],
        "overwrite": True
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Dashboard imported successfully!")
        print(f"Dashboard URL: {stack_url}/d/{response.json()['uid']}")
    else:
        print(f"Failed to import dashboard: {response.status_code} - {response.text}")

def main():
    api_key = os.getenv("GRAFANA_CLOUD_API_KEY")
    if not api_key:
        print("Please set the GRAFANA_CLOUD_API_KEY environment variable.")
        print("You can get an API key from: https://grafana.com/orgs/your-org/api-keys")
        return

    # Create the stack
    stack_data = create_grafana_stack(api_key)
    if not stack_data:
        return

    stack_slug = stack_data['slug']
    stack_url = stack_data['url']

    # Wait a bit for the stack to be fully provisioned
    print("Waiting for stack to be provisioned...")
    time.sleep(30)

    # Get stack details to confirm
    details = get_stack_details(api_key, stack_slug)
    if details:
        print(f"Stack Status: {details.get('status', 'Unknown')}")

    # Import the dashboard
    import_dashboard(stack_url, api_key, 'grafana_dashboard.json')

if __name__ == "__main__":
    main()
