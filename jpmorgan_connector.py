"""
JPMorgan API Connector - Production-Ready Flask Integration
Component 2: Flask JPMorgan Connector (Production-Ready Structure)

This module provides a comprehensive, production-ready connector for JPMorgan APIs
with OAuth token management, retry logic, error handling, and normalized responses.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import jwt
from cryptography.fernet import Fernet

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class JPMorganCredentials:
    """JPMorgan API credentials container"""
    client_id: str
    client_secret: str
    token_url: str
    base_url: str
    scope: str = "accounts:read transactions:read balances:read payments:read"

@dataclass
class TokenResponse:
    """OAuth token response structure"""
    access_token: str
    token_type: str
    expires_in: int
    scope: str
    expires_at: datetime

@dataclass
class Account:
    """Normalized account data structure"""
    id: str
    name: str
    type: str
    currency: str
    status: str
    balance: Optional[float] = None
    available_balance: Optional[float] = None

@dataclass
class Transaction:
    """Normalized transaction data structure"""
    id: str
    account_id: str
    amount: float
    currency: str
    type: str
    description: str
    timestamp: datetime
    status: str
    reference: Optional[str] = None

@dataclass
class Balance:
    """Normalized balance data structure"""
    account_id: str
    available: float
    ledger: float
    currency: str
    timestamp: datetime

class JPMorganAPIError(Exception):
    """Custom exception for JPMorgan API errors"""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}

class JPMorganConnector:
    """
    Production-ready JPMorgan API connector with comprehensive error handling,
    token management, and normalized data structures.
    """

    def __init__(self, credentials: JPMorganCredentials, encryption_key: Optional[str] = None):
        """
        Initialize the JPMorgan connector

        Args:
            credentials: JPMorgan API credentials
            encryption_key: Optional encryption key for token storage
        """
        self.credentials = credentials
        self.encryption_key = encryption_key or os.environ.get('JPMORGAN_ENCRYPTION_KEY')
        self._token: Optional[TokenResponse] = None
        self._session = self._create_session()
        self._cipher = Fernet(self.encryption_key.encode()) if self.encryption_key else None

        # API endpoints
        self.endpoints = {
            'accounts': f"{credentials.base_url}/accounts",
            'transactions': f"{credentials.base_url}/transactions",
            'balances': f"{credentials.base_url}/balances",
            'payments': f"{credentials.base_url}/payments"
        }

        logger.info("JPMorgan connector initialized successfully")

    def _create_session(self) -> requests.Session:
        """Create a configured requests session with retry logic"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            'User-Agent': 'JPMorgan-Financial-APIs/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        return session

    def _encrypt_token(self, token_data: dict) -> bytes:
        """Encrypt token data for secure storage"""
        if not self._cipher:
            return json.dumps(token_data).encode()
        return self._cipher.encrypt(json.dumps(token_data).encode())

    def _decrypt_token(self, encrypted_data: bytes) -> dict:
        """Decrypt token data"""
        if not self._cipher:
            return json.loads(encrypted_data.decode())
        return json.loads(self._cipher.decrypt(encrypted_data).decode())

    def _get_stored_token(self) -> Optional[TokenResponse]:
        """Retrieve stored token from cache/database"""
        try:
            # In a production system, this would retrieve from Redis/database
            # For now, we'll use a simple file-based approach for demonstration
            token_file = os.path.join(os.path.dirname(__file__), '.jpmorgan_token')

            if not os.path.exists(token_file):
                return None

            with open(token_file, 'rb') as f:
                encrypted_data = f.read()

            token_data = self._decrypt_token(encrypted_data)
            expires_at = datetime.fromisoformat(token_data['expires_at'])

            if expires_at <= datetime.now():
                logger.info("Stored token has expired")
                return None

            return TokenResponse(**token_data)

        except Exception as e:
            logger.warning(f"Failed to retrieve stored token: {e}")
            return None

    def _store_token(self, token: TokenResponse) -> None:
        """Store token securely"""
        try:
            token_file = os.path.join(os.path.dirname(__file__), '.jpmorgan_token')
            token_data = {
                'access_token': token.access_token,
                'token_type': token.token_type,
                'expires_in': token.expires_in,
                'scope': token.scope,
                'expires_at': token.expires_at.isoformat()
            }

            encrypted_data = self._encrypt_token(token_data)

            with open(token_file, 'wb') as f:
                f.write(encrypted_data)

            logger.info("Token stored successfully")

        except Exception as e:
            logger.error(f"Failed to store token: {e}")

    def _get_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary

        Returns:
            Valid access token string

        Raises:
            JPMorganAPIError: If token retrieval fails
        """
        # Check if we have a valid cached token
        if self._token and self._token.expires_at > datetime.now() + timedelta(minutes=5):
            return f"{self._token.token_type} {self._token.access_token}"

        # Try to load stored token
        stored_token = self._get_stored_token()
        if stored_token and stored_token.expires_at > datetime.now() + timedelta(minutes=5):
            self._token = stored_token
            return f"{stored_token.token_type} {stored_token.access_token}"

        # Need to get a new token
        try:
            logger.info("Requesting new access token from JPMorgan")

            data = {
                'grant_type': 'client_credentials',
                'client_id': self.credentials.client_id,
                'client_secret': self.credentials.client_secret,
                'scope': self.credentials.scope
            }

            response = self._session.post(
                self.credentials.token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )

            if response.status_code != 200:
                raise JPMorganAPIError(
                    f"Token request failed: {response.status_code}",
                    response.status_code,
                    response.json() if response.content else None
                )

            token_data = response.json()
            expires_at = datetime.now() + timedelta(seconds=token_data['expires_in'])

            self._token = TokenResponse(
                access_token=token_data['access_token'],
                token_type=token_data['token_type'],
                expires_in=token_data['expires_in'],
                scope=token_data.get('scope', ''),
                expires_at=expires_at
            )

            # Store the token
            self._store_token(self._token)

            logger.info("Successfully obtained new access token")
            return f"{self._token.token_type} {self._token.access_token}"

        except requests.RequestException as e:
            raise JPMorganAPIError(f"Network error during token request: {str(e)}")
        except KeyError as e:
            raise JPMorganAPIError(f"Invalid token response format: missing {e}")

    def _make_api_request(self, method: str, url: str, **kwargs) -> dict:
        """
        Make an authenticated API request with error handling

        Args:
            method: HTTP method
            url: API endpoint URL
            **kwargs: Additional request parameters

        Returns:
            API response data

        Raises:
            JPMorganAPIError: For API errors
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Get valid access token
                auth_header = self._get_access_token()
                headers = kwargs.get('headers', {})
                headers['Authorization'] = auth_header
                kwargs['headers'] = headers

                logger.debug(f"Making {method} request to {url}")

                response = self._session.request(method, url, **kwargs)

                # Handle different response codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    # Token might be invalid, clear cache and retry
                    logger.warning("Received 401, clearing token cache and retrying")
                    self._token = None
                    if os.path.exists(os.path.join(os.path.dirname(__file__), '.jpmorgan_token')):
                        os.remove(os.path.join(os.path.dirname(__file__), '.jpmorgan_token'))
                    retry_count += 1
                    continue
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                else:
                    raise JPMorganAPIError(
                        f"API request failed: {response.status_code}",
                        response.status_code,
                        response.json() if response.content else None
                    )

            except requests.RequestException as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise JPMorganAPIError(f"Network error after {max_retries} retries: {str(e)}")
                logger.warning(f"Network error, retrying ({retry_count}/{max_retries}): {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff

        raise JPMorganAPIError("Max retries exceeded")

    def get_accounts(self) -> List[Account]:
        """
        Retrieve all accounts

        Returns:
            List of normalized Account objects
        """
        try:
            logger.info("Fetching accounts from JPMorgan API")
            response_data = self._make_api_request('GET', self.endpoints['accounts'])

            accounts = []
            for account_data in response_data.get('accounts', []):
                account = Account(
                    id=account_data['id'],
                    name=account_data.get('name', ''),
                    type=account_data.get('type', ''),
                    currency=account_data.get('currency', 'USD'),
                    status=account_data.get('status', 'active'),
                    balance=account_data.get('balance'),
                    available_balance=account_data.get('available_balance')
                )
                accounts.append(account)

            logger.info(f"Successfully retrieved {len(accounts)} accounts")
            return accounts

        except Exception as e:
            logger.error(f"Failed to retrieve accounts: {e}")
            raise

    def get_transactions(self, account_id: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: int = 100) -> List[Transaction]:
        """
        Retrieve transactions with optional filtering

        Args:
            account_id: Optional account ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of transactions to retrieve

        Returns:
            List of normalized Transaction objects
        """
        try:
            params = {'limit': min(limit, 1000)}  # API limit

            if account_id:
                params['account_id'] = account_id
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()

            logger.info(f"Fetching transactions with params: {params}")
            response_data = self._make_api_request('GET', self.endpoints['transactions'], params=params)

            transactions = []
            for tx_data in response_data.get('transactions', []):
                transaction = Transaction(
                    id=tx_data['id'],
                    account_id=tx_data['account_id'],
                    amount=float(tx_data['amount']),
                    currency=tx_data.get('currency', 'USD'),
                    type=tx_data.get('type', ''),
                    description=tx_data.get('description', ''),
                    timestamp=datetime.fromisoformat(tx_data['timestamp'].replace('Z', '+00:00')),
                    status=tx_data.get('status', 'posted'),
                    reference=tx_data.get('reference')
                )
                transactions.append(transaction)

            logger.info(f"Successfully retrieved {len(transactions)} transactions")
            return transactions

        except Exception as e:
            logger.error(f"Failed to retrieve transactions: {e}")
            raise

    def get_balances(self, account_ids: Optional[List[str]] = None) -> List[Balance]:
        """
        Retrieve account balances

        Args:
            account_ids: Optional list of account IDs to filter

        Returns:
            List of normalized Balance objects
        """
        try:
            params = {}
            if account_ids:
                params['account_ids'] = ','.join(account_ids)

            logger.info("Fetching balances from JPMorgan API")
            response_data = self._make_api_request('GET', self.endpoints['balances'], params=params)

            balances = []
            for balance_data in response_data.get('balances', []):
                balance = Balance(
                    account_id=balance_data['account_id'],
                    available=float(balance_data['available']),
                    ledger=float(balance_data['ledger']),
                    currency=balance_data.get('currency', 'USD'),
                    timestamp=datetime.fromisoformat(balance_data['timestamp'].replace('Z', '+00:00'))
                )
                balances.append(balance)

            logger.info(f"Successfully retrieved {len(balances)} balances")
            return balances

        except Exception as e:
            logger.error(f"Failed to retrieve balances: {e}")
            raise

    def create_payment(self, payment_data: dict) -> dict:
        """
        Create a new payment

        Args:
            payment_data: Payment creation data

        Returns:
            Payment creation response
        """
        try:
            logger.info("Creating payment via JPMorgan API")
            response_data = self._make_api_request('POST', self.endpoints['payments'], json=payment_data)

            logger.info("Payment created successfully")
            return response_data

        except Exception as e:
            logger.error(f"Failed to create payment: {e}")
            raise

    def get_connection_status(self) -> dict:
        """
        Check the connection status to JPMorgan APIs

        Returns:
            Status information dictionary
        """
        try:
            # Try to get a token to test connectivity
            token = self._get_access_token()

            return {
                'status': 'connected',
                'timestamp': datetime.now().isoformat(),
                'token_valid': True,
                'message': 'Successfully connected to JPMorgan APIs'
            }

        except JPMorganAPIError as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'token_valid': False,
                'error': str(e),
                'message': 'Failed to connect to JPMorgan APIs'
            }

        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'token_valid': False,
                'error': str(e),
                'message': 'Unexpected error during connection test'
            }

# Factory function for easy initialization
def create_jpmorgan_connector() -> JPMorganConnector:
    """
    Create a JPMorgan connector instance using environment variables

    Returns:
        Configured JPMorganConnector instance
    """
    credentials = JPMorganCredentials(
        client_id=os.environ.get('JPMORGAN_CLIENT_ID', ''),
        client_secret=os.environ.get('JPMORGAN_CLIENT_SECRET', ''),
        token_url=os.environ.get('JPMORGAN_TOKEN_URL', ''),
        base_url=os.environ.get('JPMORGAN_BASE_URL', ''),
        scope=os.environ.get('JPMORGAN_SCOPE', 'accounts:read transactions:read balances:read payments:read')
    )

    encryption_key = os.environ.get('JPMORGAN_ENCRYPTION_KEY')

    if not all([credentials.client_id, credentials.client_secret, credentials.token_url, credentials.base_url]):
        raise ValueError("Missing required JPMorgan API credentials in environment variables")

    return JPMorganConnector(credentials, encryption_key)
