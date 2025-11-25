"""
Comprehensive Test Suite for JPMorgan API Endpoints
Tests OpenBanking and API Gateway across multiple environments
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Set testing environment
os.environ['TESTING'] = '1'
os.environ['ALLOW_MISSING_TOKENS'] = 'true'

from src.jpmorgan_client import JPMorganAPIClient, get_jpmorgan_client


class TestJPMorganClientInitialization:
    """Test client initialization and configuration"""

    def test_client_initialization_production(self):
        """Test client initializes with production environment"""
        client = JPMorganAPIClient(environment="production")
        assert client.environment == "production"
        assert client.openbanking_production_url == (
            "https://openbanking.jpmorgan.com/accessapi"
        )
        assert client.apigateway_production_url == (
            "https://apigateway.jpmorgan.com/accessapi"
        )

    def test_client_initialization_uat(self):
        """Test client initializes with UAT environment"""
        client = JPMorganAPIClient(environment="uat")
        assert client.environment == "uat"
        assert client.openbanking_uat_url == (
            "https://openbankinguat.jpmorgan.com/accessapi"
        )

    def test_client_initialization_qaf(self):
        """Test client initializes with QAF environment"""
        client = JPMorganAPIClient(environment="qaf")
        assert client.environment == "qaf"
        assert client.apigateway_qaf_url == (
            "https://apigatewayqaf.jpmorgan.com/accessapi"
        )

    def test_get_service_url_openbanking_production(self):
        """Test getting OpenBanking URL for production"""
        client = JPMorganAPIClient(environment="production")
        url = client.get_service_url("openbanking")
        assert url == "https://openbanking.jpmorgan.com/accessapi"

    def test_get_service_url_openbanking_uat(self):
        """Test getting OpenBanking URL for UAT"""
        client = JPMorganAPIClient(environment="uat")
        url = client.get_service_url("openbanking")
        assert url == "https://openbankinguat.jpmorgan.com/accessapi"

    def test_get_service_url_apigateway_production(self):
        """Test getting API Gateway URL for production"""
        client = JPMorganAPIClient(environment="production")
        url = client.get_service_url("apigateway")
        assert url == "https://apigateway.jpmorgan.com/accessapi"

    def test_get_service_url_apigateway_qaf(self):
        """Test getting API Gateway URL for QAF"""
        client = JPMorganAPIClient(environment="qaf")
        url = client.get_service_url("apigateway")
        assert url == "https://apigatewayqaf.jpmorgan.com/accessapi"

    def test_get_service_url_invalid_service(self):
        """Test error handling for invalid service"""
        client = JPMorganAPIClient(environment="production")
        with pytest.raises(ValueError, match="Unknown service"):
            client.get_service_url("invalid_service")

    def test_set_environment(self):
        """Test environment switching"""
        client = JPMorganAPIClient(environment="production")
        assert client.environment == "production"

        client.set_environment("uat")
        assert client.environment == "uat"

        client.set_environment("qaf")
        assert client.environment == "qaf"

    def test_singleton_pattern(self):
        """Test singleton pattern with environment support"""
        client1 = get_jpmorgan_client("production")
        client2 = get_jpmorgan_client("production")
        assert client1 is client2

        # Different environment creates new instance
        client3 = get_jpmorgan_client("uat")
        assert client3 is not client1


class TestOpenBankingAPI:
    """Test OpenBanking API methods"""

    @pytest.mark.asyncio
    async def test_openbanking_health_check_success(self):
        """Test successful OpenBanking health check"""
        client = JPMorganAPIClient(environment="production")

        # Mock the HTTP client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await client.openbanking_health_check()

            assert result["status"] == "healthy"
            assert result["environment"] == "production"
            assert "timestamp" in result
            mock_get.assert_called_once()

        await client.close()

    @pytest.mark.asyncio
    async def test_openbanking_health_check_failure(self):
        """Test OpenBanking health check failure"""
        client = JPMorganAPIClient(environment="production")

        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPError("Connection failed")

            result = await client.openbanking_health_check()

            assert result["status"] == "unhealthy"
            assert "error" in result
            assert result["environment"] == "production"

        await client.close()

    @pytest.mark.asyncio
    async def test_openbanking_get_accounts(self):
        """Test getting accounts from OpenBanking API"""
        client = JPMorganAPIClient(environment="production")

        # Mock access token
        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"

            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "accounts": [
                    {"account_id": "ACC001", "account_type": "checking"},
                    {"account_id": "ACC002", "account_type": "savings"}
                ]
            }

            with patch.object(
                client.client, 'get', new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_response

                accounts = await client.openbanking_get_accounts(user_id="user123")

                assert len(accounts) == 2
                assert accounts[0]["account_id"] == "ACC001"
                assert accounts[1]["account_type"] == "savings"
                mock_token.assert_called_once_with("openbanking")

        await client.close()

    @pytest.mark.asyncio
    async def test_openbanking_get_transactions(self):
        """Test getting transactions from OpenBanking API"""
        client = JPMorganAPIClient(environment="production")

        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "transactions": [
                    {"transaction_id": "TXN001", "amount": 100.00},
                    {"transaction_id": "TXN002", "amount": 250.50}
                ]
            }

            with patch.object(
                client.client, 'get', new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_response

                transactions = await client.openbanking_get_transactions(
                    account_id="ACC001",
                    start_date="2024-01-01",
                    end_date="2024-12-31"
                )

                assert len(transactions) == 2
                assert transactions[0]["transaction_id"] == "TXN001"
                assert transactions[1]["amount"] == 250.50

        await client.close()

    @pytest.mark.asyncio
    async def test_openbanking_get_balance(self):
        """Test getting account balance from OpenBanking API"""
        client = JPMorganAPIClient(environment="production")

        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "account_id": "ACC001",
                "balance": 5000.00,
                "currency": "USD"
            }

            with patch.object(
                client.client, 'get', new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_response

                balance = await client.openbanking_get_balance(account_id="ACC001")

                assert balance["account_id"] == "ACC001"
                assert balance["balance"] == 5000.00
                assert balance["currency"] == "USD"

        await client.close()


class TestAPIGateway:
    """Test API Gateway methods"""

    @pytest.mark.asyncio
    async def test_apigateway_health_check_success(self):
        """Test successful API Gateway health check"""
        client = JPMorganAPIClient(environment="production")

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            result = await client.apigateway_health_check()

            assert result["status"] == "healthy"
            assert result["environment"] == "production"
            assert "timestamp" in result

        await client.close()

    @pytest.mark.asyncio
    async def test_apigateway_execute_request(self):
        """Test executing request through API Gateway"""
        client = JPMorganAPIClient(environment="production")

        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "status": "success",
                "data": {"key": "value"}
            }

            with patch.object(
                client.client, 'request', new_callable=AsyncMock
            ) as mock_request:
                mock_request.return_value = mock_response

                result = await client.apigateway_execute_request(
                    method="GET",
                    endpoint="/v1/test",
                    params={"param1": "value1"}
                )

                assert result["status"] == "success"
                assert result["data"]["key"] == "value"
                mock_token.assert_called_once_with("apigateway")

        await client.close()

    @pytest.mark.asyncio
    async def test_apigateway_get_services(self):
        """Test getting available services from API Gateway"""
        client = JPMorganAPIClient(environment="production")

        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "services": [
                    {"name": "payments", "status": "active"},
                    {"name": "accounts", "status": "active"}
                ]
            }

            with patch.object(
                client.client, 'get', new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_response

                services = await client.apigateway_get_services()

                assert len(services) == 2
                assert services[0]["name"] == "payments"
                assert services[1]["status"] == "active"

        await client.close()

    @pytest.mark.asyncio
    async def test_apigateway_get_api_status(self):
        """Test getting API Gateway status and metrics"""
        client = JPMorganAPIClient(environment="production")

        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"
            
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "uptime": 99.99,
                "request_count": 1000000,
                "error_rate": 0.01
            }

            with patch.object(
                client.client, 'get', new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_response

                status = await client.apigateway_get_api_status()

                assert status["uptime"] == 99.99
                assert status["request_count"] == 1000000
                assert status["error_rate"] == 0.01

        await client.close()


class TestMultiEnvironment:
    """Test multi-environment functionality"""

    @pytest.mark.asyncio
    async def test_production_environment(self):
        """Test production environment configuration"""
        client = JPMorganAPIClient(environment="production")

        assert client.environment == "production"
        assert client.get_service_url("openbanking") == (
            "https://openbanking.jpmorgan.com/accessapi"
        )
        assert client.get_service_url("apigateway") == (
            "https://apigateway.jpmorgan.com/accessapi"
        )

        await client.close()

    @pytest.mark.asyncio
    async def test_uat_environment(self):
        """Test UAT environment configuration"""
        client = JPMorganAPIClient(environment="uat")

        assert client.environment == "uat"
        assert client.get_service_url("openbanking") == (
            "https://openbankinguat.jpmorgan.com/accessapi"
        )
        # UAT uses production API Gateway
        assert client.get_service_url("apigateway") == (
            "https://apigateway.jpmorgan.com/accessapi"
        )

        await client.close()

    @pytest.mark.asyncio
    async def test_qaf_environment(self):
        """Test QAF environment configuration"""
        client = JPMorganAPIClient(environment="qaf")

        assert client.environment == "qaf"
        # QAF uses production OpenBanking
        assert client.get_service_url("openbanking") == (
            "https://openbanking.jpmorgan.com/accessapi"
        )
        assert client.get_service_url("apigateway") == (
            "https://apigatewayqaf.jpmorgan.com/accessapi"
        )

        await client.close()

    @pytest.mark.asyncio
    async def test_environment_switching(self):
        """Test switching between environments"""
        client = JPMorganAPIClient(environment="production")

        # Start with production
        assert client.environment == "production"
        prod_url = client.get_service_url("openbanking")
        assert "openbanking.jpmorgan.com" in prod_url

        # Switch to UAT
        client.set_environment("uat")
        assert client.environment == "uat"
        uat_url = client.get_service_url("openbanking")
        assert "openbankinguat.jpmorgan.com" in uat_url

        # Switch to QAF
        client.set_environment("qaf")
        assert client.environment == "qaf"
        qaf_url = client.get_service_url("apigateway")
        assert "apigatewayqaf.jpmorgan.com" in qaf_url

        await client.close()


class TestErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_missing_credentials(self):
        """Test handling of missing credentials"""
        client = JPMorganAPIClient(environment="production")

        # Clear credentials
        client.projects["openbanking"]["client_id"] = None

        with pytest.raises(ValueError, match="Missing credentials"):
            await client.get_access_token("openbanking")

        await client.close()

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors"""
        client = JPMorganAPIClient(environment="production")

        # Mock the access token first
        with patch.object(
            client, 'get_access_token', new_callable=AsyncMock
        ) as mock_token:
            mock_token.return_value = "mock_access_token"

            with patch.object(
                client.client, 'get', new_callable=AsyncMock
            ) as mock_get:
                mock_get.side_effect = httpx.HTTPError("Network error")
                
                # Should return empty list on error
                accounts = await client.openbanking_get_accounts(user_id="user123")
                assert accounts == []
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_http_error_handling(self):
        """Test handling of HTTP errors"""
        client = JPMorganAPIClient(environment="production")
        
        with patch.object(client, 'get_access_token', new_callable=AsyncMock) as mock_token:
            mock_token.return_value = "mock_access_token"
            
            import httpx
            with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = httpx.HTTPError("HTTP 500 Error")
                
                # Should return empty dict on error
                balance = await client.openbanking_get_balance(account_id="ACC001")
                assert balance == {}
        
        await client.close()


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_openbanking_workflow(self):
        """Test complete OpenBanking workflow"""
        client = JPMorganAPIClient(environment="production")
        
        with patch.object(client, 'get_access_token', new_callable=AsyncMock) as mock_token:
            mock_token.return_value = "mock_access_token"
            
            # Mock health check
            health_response = MagicMock()
            health_response.raise_for_status = MagicMock()
            
            # Mock accounts response
            accounts_response = MagicMock()
            accounts_response.raise_for_status = MagicMock()
            accounts_response.json.return_value = {
                "accounts": [{"account_id": "ACC001", "account_type": "checking"}]
            }
            
            # Mock balance response
            balance_response = MagicMock()
            balance_response.raise_for_status = MagicMock()
            balance_response.json.return_value = {
                "account_id": "ACC001",
                "balance": 5000.00,
                "currency": "USD"
            }
            
            with patch.object(client.client, 'get', new_callable=AsyncMock) as mock_get:
                mock_get.side_effect = [health_response, accounts_response, balance_response]
                
                # 1. Health check
                health = await client.openbanking_health_check()
                assert health["status"] == "healthy"
                
                # 2. Get accounts
                accounts = await client.openbanking_get_accounts(user_id="user123")
                assert len(accounts) == 1
                
                # 3. Get balance
                balance = await client.openbanking_get_balance(account_id="ACC001")
                assert balance["balance"] == 5000.00
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_multi_environment_workflow(self):
        """Test workflow across multiple environments"""
        # Test in production
        prod_client = get_jpmorgan_client("production")
        assert prod_client.environment == "production"
        
        # Test in UAT
        uat_client = get_jpmorgan_client("uat")
        assert uat_client.environment == "uat"
        assert uat_client is not prod_client
        
        # Test in QAF
        qaf_client = get_jpmorgan_client("qaf")
        assert qaf_client.environment == "qaf"
        assert qaf_client is not prod_client
        assert qaf_client is not uat_client
        
        await prod_client.close()
        await uat_client.close()
        await qaf_client.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
