"""
JP Morgan Payments API Client
Integrates with JP Morgan Developer Portal APIs
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import httpx
import structlog

logger = structlog.get_logger()


class JPMorganAPIClient:
    """Client for JP Morgan Payments APIs"""

    def __init__(self) -> None:
        """Initialize JP Morgan API client"""
        self.base_url = os.getenv(
            "JPMORGAN_BASE_URL",
            "https://api.payments.jpmorgan.com"
        )
        self.auth_url = os.getenv(
            "JPMORGAN_AUTH_URL",
            "https://auth.payments.jpmorgan.com"
        )


        # Project credentials
        self.projects = {
            "ai_accounts": {
                "client_id": os.getenv("JPMORGAN_AI_ACCOUNTS_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_AI_ACCOUNTS_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_AI_ACCOUNTS_API_KEY"),
            },
            "corporate_login": {
                "client_id": os.getenv("JPMORGAN_CORPORATE_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_CORPORATE_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_CORPORATE_API_KEY"),
            },
            "payroll": {
                "client_id": os.getenv("JPMORGAN_PAYROLL_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_PAYROLL_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_PAYROLL_API_KEY"),
            },
            "petty_cash": {
                "client_id": os.getenv("JPMORGAN_PETTY_CASH_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_PETTY_CASH_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_PETTY_CASH_API_KEY"),
            },
            "owl1": {
                "client_id": os.getenv("JPMORGAN_OWL1_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_OWL1_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_OWL1_API_KEY"),
            }
        }


        # Token cache
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.client = httpx.AsyncClient(timeout=30.0)


    async def get_access_token(self, project: str) -> str:
        """Get OAuth access token for a project"""
        try:
            # Check if we have a valid cached token
            if project in self.tokens:
                token_data = self.tokens[project]
                if datetime.now() < token_data["expires_at"]:
                    return str(token_data["access_token"])


            # Get new token
            credentials = self.projects.get(project)
            if not credentials or not credentials["client_id"]:
                raise ValueError(f"Missing credentials for project: {project}")


            response = await self.client.post(
                f"{self.auth_url}/oauth/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": credentials["client_id"],
                    "client_secret": credentials["client_secret"],
                    "scope": "payments"
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            response.raise_for_status()


            token_data = response.json()
            access_token = str(token_data["access_token"])
            expires_in = int(token_data.get("expires_in", 3600))


            # Cache token
            self.tokens[project] = {
                "access_token": access_token,
                "expires_at": datetime.now() + timedelta(seconds=expires_in - 60)
            }


            logger.info("Obtained access token", project=project)
            return access_token


        except httpx.HTTPError as e:
            logger.error("Failed to get access token", project=project, error=str(e))
            raise
        except (KeyError, ValueError) as e:
            logger.error("Invalid token response", project=project, error=str(e))
            raise


    async def _make_request(
        self,
        method: str,
        endpoint: str,
        project: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to JP Morgan API"""
        try:
            access_token = await self.get_access_token(project)
            credentials = self.projects[project]


            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": credentials["api_key"],
                "Content-Type": "application/json",
                "Accept": "application/json"
            }


            url = f"{self.base_url}{endpoint}"


            response = await self.client.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params
            )
            response.raise_for_status()


            return response.json()


        except httpx.HTTPStatusError as e:
            logger.error(
                "API request failed",
                project=project,
                endpoint=endpoint,
                status=e.response.status_code,
                error=str(e)
            )
            raise
        except httpx.HTTPError as e:
            logger.error(
                "Request error",
                project=project,
                endpoint=endpoint,
                error=str(e)
            )
            raise


    # AI ACCOUNTS APIs
    async def get_accounts(self, account_type: str = "all") -> List[Dict[str, Any]]:
        """Get accounts from AI ACCOUNTS project


        Args:
            account_type: 'corporate', 'business', 'personal', or 'all'
        """
        try:
            params: Dict[str, str] = {}
            if account_type != "all":
                params["type"] = account_type


            result = await self._make_request(
                "GET",
                "/v1/accounts",
                "ai_accounts",
                params=params
            )

            logger.info(
                "Retrieved accounts",
                account_type=account_type,
                count=len(result.get("accounts", []))
            )
            return result.get("accounts", [])

        except httpx.HTTPError as e:
            logger.error("Failed to get accounts", error=str(e))
            return []


    async def get_account_balance(self, account_id: str) -> Dict[str, Any]:
        """Get account balance"""
        try:
            result = await self._make_request(
                "GET",
                f"/v1/accounts/{account_id}/balance",
                "ai_accounts"
            )
            return result
        except httpx.HTTPError as e:
            logger.error(
                "Failed to get account balance",
                account_id=account_id,
                error=str(e)
            )
            return {}


    async def get_account_transactions(
        self,
        account_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get account transactions"""
        try:
            params: Dict[str, Any] = {"limit": limit}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date


            result = await self._make_request(
                "GET",
                f"/v1/accounts/{account_id}/transactions",
                "ai_accounts",
                params=params
            )
            return result.get("transactions", [])
        except httpx.HTTPError as e:
            logger.error(
                "Failed to get transactions",
                account_id=account_id,
                error=str(e)
            )
            return []


    # CORPORATE EXECUTIVE LOGIN APIs
    async def corporate_login(self, username: str, password: str) -> Dict[str, Any]:
        """Corporate executive login"""
        try:
            result = await self._make_request(
                "POST",
                "/v1/auth/corporate/login",
                "corporate_login",
                data={
                    "username": username,
                    "password": password
                }
            )
            logger.info("Corporate login successful", username=username)
            return result
        except httpx.HTTPError as e:
            logger.error("Corporate login failed", username=username, error=str(e))
            raise


    async def get_corporate_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get corporate user information"""
        try:
            result = await self._make_request(
                "GET",
                f"/v1/users/corporate/{user_id}",
                "corporate_login"
            )
            return result
        except httpx.HTTPError as e:
            logger.error(
                "Failed to get corporate user info",
                user_id=user_id,
                error=str(e)
            )
            return {}


    # OWL PAYROLL APIs
    async def get_payroll_data(
        self,
        employee_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get payroll data"""
        try:
            params: Dict[str, str] = {}
            if employee_id:
                params["employee_id"] = employee_id
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date


            result = await self._make_request(
                "GET",
                "/v1/payroll",
                "payroll",
                params=params
            )
            return result.get("payroll_records", [])
        except httpx.HTTPError as e:
            logger.error("Failed to get payroll data", error=str(e))
            return []


    async def process_payroll(self, payroll_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process payroll payment"""
        try:
            result = await self._make_request(
                "POST",
                "/v1/payroll/process",
                "payroll",
                data=payroll_data
            )
            logger.info("Payroll processed successfully")
            return result
        except httpx.HTTPError as e:
            logger.error("Failed to process payroll", error=str(e))
            raise


    # OWL PETTY CASH APIs
    async def get_petty_cash_balance(self) -> Dict[str, Any]:
        """Get petty cash balance"""
        try:
            result = await self._make_request(
                "GET",
                "/v1/petty-cash/balance",
                "petty_cash"
            )
            return result
        except httpx.HTTPError as e:
            logger.error("Failed to get petty cash balance", error=str(e))
            return {}


    async def create_petty_cash_request(
        self,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create petty cash request"""
        try:
            result = await self._make_request(
                "POST",
                "/v1/petty-cash/requests",
                "petty_cash",
                data=request_data
            )
            logger.info("Petty cash request created")
            return result
        except httpx.HTTPError as e:
            logger.error("Failed to create petty cash request", error=str(e))
            raise


    async def get_petty_cash_transactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get petty cash transactions"""
        try:
            params: Dict[str, str] = {}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

            result = await self._make_request(
                "GET",
                "/v1/petty-cash/transactions",
                "petty_cash",
                params=params
            )
            return result.get("transactions", [])
        except httpx.HTTPError as e:
            logger.error("Failed to get petty cash transactions", error=str(e))
            return []


    # Owl1 Data Integration APIs
    async def sync_data(self, data_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync data with Owl1 integration"""
        try:
            result = await self._make_request(
                "POST",
                f"/v1/integration/sync/{data_type}",
                "owl1",
                data=data
            )
            logger.info("Data synced successfully", data_type=data_type)
            return result
        except httpx.HTTPError as e:
            logger.error("Failed to sync data", data_type=data_type, error=str(e))
            raise


    async def get_integration_status(self) -> Dict[str, Any]:
        """Get Owl1 integration status"""
        try:
            result = await self._make_request(
                "GET",
                "/v1/integration/status",
                "owl1"
            )
            return result
        except httpx.HTTPError as e:
            logger.error("Failed to get integration status", error=str(e))
            return {}

    async def close(self) -> None:
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance
_JPMORGAN_CLIENT: Optional[JPMorganAPIClient] = None


def get_jpmorgan_client() -> JPMorganAPIClient:
    """Get JP Morgan API client instance"""
    global _JPMORGAN_CLIENT
    if _JPMORGAN_CLIENT is None:
        _JPMORGAN_CLIENT = JPMorganAPIClient()
    return _JPMORGAN_CLIENT
