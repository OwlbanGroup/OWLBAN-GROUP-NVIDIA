"""
Test JP Morgan API Connection
Quick test to verify credentials and API connectivity
"""
import asyncio
import os
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv('.env.jpmorgan')

async def test_jpmorgan_connection():
    """Test connection to JP Morgan API"""

    print("=" * 60)
    print("JP MORGAN API CONNECTION TEST")
    print("=" * 60)
    print()

    # Get credentials
    auth_url = os.getenv("JPMORGAN_AUTH_URL", "https://id.payments.jpmorgan.com/am/oauth2/alpha")
    client_id = os.getenv("JPMORGAN_CLIENT_ID")
    client_secret = os.getenv("JPMORGAN_CLIENT_SECRET")

    print(f"Auth URL: {auth_url}")
    print(f"Client ID: {client_id[:20]}...")
    print(f"Client Secret: {client_secret[:20]}...")
    print()

    if not client_id or not client_secret:
        print("❌ ERROR: Credentials not found in .env.jpmorgan")
        return False

    print("Testing OAuth token retrieval...")
    print("-" * 60)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try different scope configurations
            scope_options = [
                None,  # No scope
                "",    # Empty scope
                "openid",  # OpenID scope
                "api",  # Generic API scope
            ]

            for scope in scope_options:
                print(f"\nTrying with scope: {scope if scope else 'None'}")

                data = {
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                }

                if scope:
                    data["scope"] = scope

                response = await client.post(
                    f"{auth_url}/access_token",
                    data=data,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded"
                    }
                )

                if response.status_code == 200:
                    break

            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            print()

            if response.status_code == 200:
                token_data = response.json()
                print("✅ SUCCESS! Token obtained successfully")
                print(f"Access Token: {token_data.get('access_token', '')[:30]}...")
                print(f"Token Type: {token_data.get('token_type', 'N/A')}")
                print(f"Expires In: {token_data.get('expires_in', 'N/A')} seconds")
                print()
                print("=" * 60)
                print("🎉 JP MORGAN API CONNECTION SUCCESSFUL!")
                print("=" * 60)
                return True
            else:
                print(f"❌ ERROR: Failed to get token")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                return False

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print()
        print("Possible issues:")
        print("1. Check if credentials are correct")
        print("2. Verify network connectivity")
        print("3. Confirm JP Morgan API endpoint is accessible")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_jpmorgan_connection())
    exit(0 if result else 1)
