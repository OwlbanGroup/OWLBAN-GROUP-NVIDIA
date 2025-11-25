"""
JP Morgan Payments API Client
Integrates with JP Morgan Developer Portal APIs
Supports multiple environments: Production, UAT, QAF
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import httpx
import structlog

logger = structlog.get_logger()


class JPMorganAPIClient:
    """Client for JP Morgan Payments APIs
    
    Supports multiple environments and services:
    - Production OpenBanking: https://openbanking.jpmorgan.com/accessapi
    - Production API Gateway: https://apigateway.jpmorgan.com/accessapi
    - UAT OpenBanking: https://openbankinguat.jpmorgan.com/accessapi
    - QAF API Gateway: https://apigatewayqaf.jpmorgan.com/accessapi
    """

    def __init__(self, environment: str = "production") -> None:
        """Initialize JP Morgan API client
        
        Args:
            environment: 'production', 'uat', or 'qaf'
        """
        self.environment = environment.lower()
        
        # Legacy base URLs
        self.base_url = os.getenv(
            "JPMORGAN_BASE_URL",
            "https://api.payments.jpmorgan.com"
        )
        self.auth_url = os.getenv(
            "JPMORGAN_AUTH_URL",
            "https://auth.payments.jpmorgan.com"
        )
        
        # New endpoint URLs
        self.openbanking_production_url = os.getenv(
            "JPMORGAN_OPENBANKING_PRODUCTION_URL",
            "https://openbanking.jpmorgan.com/accessapi"
        )
        self.openbanking_uat_url = os.getenv(
            "JPMORGAN_OPENBANKING_UAT_URL",
            "https://openbankinguat.jpmorgan.com/accessapi"
        )
        self.apigateway_production_url = os.getenv(
            "JPMORGAN_APIGATEWAY_PRODUCTION_URL",
            "https://apigateway.jpmorgan.com/accessapi"
        )
        self.apigateway_qaf_url = os.getenv(
            "JPMORGAN_APIGATEWAY_QAF_URL",
            "https://apigatewayqaf.jpmorgan.com/accessapi"
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
            },
            "openbanking": {
                "client_id": os.getenv("JPMORGAN_OPENBANKING_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_OPENBANKING_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_OPENBANKING_API_KEY"),
            },
            "apigateway": {
                "client_id": os.getenv("JPMORGAN_APIGATEWAY_CLIENT_ID"),
                "client_secret": os.getenv("JPMORGAN_APIGATEWAY_CLIENT_SECRET"),
                "api_key": os.getenv("JPMORGAN_APIGATEWAY_API_KEY"),
            }
        }

        # Token cache
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        
        logger.info(
            "JPMorgan API Client initialized",
            environment=self.environment,
            openbanking_url=self.get_service_url("openbanking"),
            apigateway_url=self.get_service_url("apigateway")
        )
    
    def get_service_url(self, service: str) -> str:
        """Get the appropriate service URL based on environment
        
        Args:
            service: 'openbanking' or 'apigateway'
            
        Returns:
            The service URL for the current environment
        """
        if service == "openbanking":
            if self.environment == "uat":
                return self.openbanking_uat_url
            else:  # production (default)
                return self.openbanking_production_url
        elif service == "apigateway":
            if self.environment == "qaf":
                return self.apigateway_qaf_url
            else:  # production (default)
                return self.apigateway_production_url
        else:
            raise ValueError(f"Unknown service: {service}")
    
    def set_environment(self, environment: str) -> None:
        """Change the environment
        
        Args:
            environment: 'production', 'uat', or 'qaf'
        """
        self.environment = environment.lower()
        logger.info("Environment changed", new_environment=self.environment)


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

    # OpenBanking API Methods
    async def openbanking_health_check(self) -> Dict[str, Any]:
        """Check OpenBanking API health status"""
        try:
            url = self.get_service_url("openbanking")
            response = await self.client.get(f"{url}/health")
            response.raise_for_status()
            
            logger.info("OpenBanking health check successful", environment=self.environment)
            return {
                "status": "healthy",
                "environment": self.environment,
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
        except httpx.HTTPError as e:
            logger.error("OpenBanking health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "environment": self.environment,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def openbanking_get_accounts(self, user_id: str) -> List[Dict[str, Any]]:
        """Get accounts from OpenBanking API
        
        Args:
            user_id: User identifier
            
        Returns:
            List of account information
        """
        try:
            access_token = await self.get_access_token("openbanking")
            url = self.get_service_url("openbanking")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": self.projects["openbanking"]["api_key"],
                "Content-Type": "application/json"
            }
            
            response = await self.client.get(
                f"{url}/accounts",
                headers=headers,
                params={"user_id": user_id}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Retrieved OpenBanking accounts", user_id=user_id, count=len(result.get("accounts", [])))
            return result.get("accounts", [])
            
        except httpx.HTTPError as e:
            logger.error("Failed to get OpenBanking accounts", user_id=user_id, error=str(e))
            return []
    
    async def openbanking_get_transactions(
        self,
        account_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get transactions from OpenBanking API
        
        Args:
            account_id: Account identifier
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            List of transactions
        """
        try:
            access_token = await self.get_access_token("openbanking")
            url = self.get_service_url("openbanking")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": self.projects["openbanking"]["api_key"],
                "Content-Type": "application/json"
            }
            
            params: Dict[str, str] = {}
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date
            
            response = await self.client.get(
                f"{url}/accounts/{account_id}/transactions",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Retrieved OpenBanking transactions", account_id=account_id, count=len(result.get("transactions", [])))
            return result.get("transactions", [])
            
        except httpx.HTTPError as e:
            logger.error("Failed to get OpenBanking transactions", account_id=account_id, error=str(e))
            return []
    
    async def openbanking_get_balance(self, account_id: str) -> Dict[str, Any]:
        """Get account balance from OpenBanking API
        
        Args:
            account_id: Account identifier
            
        Returns:
            Balance information
        """
        try:
            access_token = await self.get_access_token("openbanking")
            url = self.get_service_url("openbanking")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": self.projects["openbanking"]["api_key"],
                "Content-Type": "application/json"
            }
            
            response = await self.client.get(
                f"{url}/accounts/{account_id}/balance",
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Retrieved OpenBanking balance", account_id=account_id)
            return result
            
        except httpx.HTTPError as e:
            logger.error("Failed to get OpenBanking balance", account_id=account_id, error=str(e))
            return {}
    
    # API Gateway Methods
    async def apigateway_health_check(self) -> Dict[str, Any]:
        """Check API Gateway health status"""
        try:
            url = self.get_service_url("apigateway")
            response = await self.client.get(f"{url}/health")
            response.raise_for_status()
            
            logger.info("API Gateway health check successful", environment=self.environment)
            return {
                "status": "healthy",
                "environment": self.environment,
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
        except httpx.HTTPError as e:
            logger.error("API Gateway health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "environment": self.environment,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def apigateway_execute_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a request through API Gateway
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            API response
        """
        try:
            access_token = await self.get_access_token("apigateway")
            url = self.get_service_url("apigateway")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": self.projects["apigateway"]["api_key"],
                "Content-Type": "application/json"
            }
            
            response = await self.client.request(
                method=method,
                url=f"{url}{endpoint}",
                headers=headers,
                json=data,
                params=params
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("API Gateway request successful", method=method, endpoint=endpoint)
            return result
            
        except httpx.HTTPError as e:
            logger.error("API Gateway request failed", method=method, endpoint=endpoint, error=str(e))
            raise
    
    async def apigateway_get_services(self) -> List[Dict[str, Any]]:
        """Get list of available services from API Gateway
        
        Returns:
            List of available services
        """
        try:
            access_token = await self.get_access_token("apigateway")
            url = self.get_service_url("apigateway")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": self.projects["apigateway"]["api_key"],
                "Content-Type": "application/json"
            }
            
            response = await self.client.get(
                f"{url}/services",
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Retrieved API Gateway services", count=len(result.get("services", [])))
            return result.get("services", [])
            
        except httpx.HTTPError as e:
            logger.error("Failed to get API Gateway services", error=str(e))
            return []
    
    async def apigateway_get_api_status(self) -> Dict[str, Any]:
        """Get API Gateway status and metrics
        
        Returns:
            Status and metrics information
        """
        try:
            access_token = await self.get_access_token("apigateway")
            url = self.get_service_url("apigateway")
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "X-API-Key": self.projects["apigateway"]["api_key"],
                "Content-Type": "application/json"
            }
            
            response = await self.client.get(
                f"{url}/status",
                headers=headers
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info("Retrieved API Gateway status")
            return result
            
        except httpx.HTTPError as e:
            logger.error("Failed to get API Gateway status", error=str(e))
            return {}

    async def close(self) -> None:
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance with environment support
_JPMORGAN_CLIENT: Optional[JPMorganAPIClient] = None


def get_jpmorgan_client(environment: str = "production") -> JPMorganAPIClient:
    """Get JP Morgan API client instance
    
    Args:
        environment: 'production', 'uat', or 'qaf'
        
    Returns:
        JPMorganAPIClient instance
    """
    global _JPMORGAN_CLIENT
    if _JPMORGAN_CLIENT is None:
        _JPMORGAN_CLIENT = JPMorganAPIClient(environment=environment)
    elif _JPMORGAN_CLIENT.environment != environment.lower():
        # Create new instance if environment changed
        _JPMORGAN_CLIENT = JPMorganAPIClient(environment=environment)
    return _JPMORGAN_CLIENT
