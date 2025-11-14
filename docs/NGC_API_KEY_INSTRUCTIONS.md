# How to Obtain NVIDIA NGC Personal Access Token (API Key)

To use NVIDIA NGC services such as containers, models, and endpoints, you need to generate a personal access token (API key) from your NVIDIA NGC account.

## Steps to Get Your NGC API Key

1. Go to the NVIDIA NGC website: https://ngc.nvidia.com/
2. Log in with your NVIDIA account credentials.
3. Click on your profile icon or username in the top right corner.
4. Select **API Key** or **API Tokens** from the dropdown menu.
5. If you do not have an existing API key, click **Create API Key**.
6. Copy the generated API key.

## How to Use the API Key in This Project

- Set the API key as an environment variable named `NGC_API_KEY`.
- You can add it to your `.env` file in the project root:

```
NGC_API_KEY=your_personal_access_token_here
```

- The deployment script (`deploy_local.sh`) will automatically detect this environment variable and configure the NVIDIA NGC CLI with your API key.

## Additional Notes

- Keep your API key secure and do not share it publicly.
- If you lose your API key, you can revoke it and generate a new one from the NVIDIA NGC website.
