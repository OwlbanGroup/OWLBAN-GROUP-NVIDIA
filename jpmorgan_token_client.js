import axios from "axios";
import qs from "qs";
import dotenv from "dotenv";

dotenv.config();

async function getJpmAccessToken() {
  // SECURITY WARNING: Credentials have been exposed and need rotation
  // TODO: Rotate JPMorgan API credentials immediately
  // 1. Log into JPMorgan Developer Portal
  // 2. Delete the exposed Client Secret
  // 3. Generate new Client ID and Client Secret
  // 4. Update environment variables with new credentials

  const data = qs.stringify({
    grant_type: "client_credentials",
    scope: "jpm:payments:sandbox"
  });

  const response = await axios.post(
    process.env.JPM_TOKEN_URL,
    data,
    {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      auth: {
        username: process.env.JPM_CLIENT_ID,
        password: process.env.JPM_CLIENT_SECRET
      }
    }
  );

  return response.data.access_token;
}

export default getJpmAccessToken;
