"""
Apollo.io Data Enrichment Connector - Production-Ready Flask Integration
Component: Apollo.io Enrichment Connector

This module provides a comprehensive, production-ready connector for Apollo.io APIs
with authentication, retry logic, error handling, and normalized enrichment responses.
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

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ApolloCredentials:
    """Apollo.io API credentials container"""
    api_key: str
    base_url: str = "https://api.apollo.io/v1"

@dataclass
class EnrichmentRequest:
    """Enrichment request structure"""
    email: Optional[str] = None
    domain: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company_name: Optional[str] = None
    linkedin_url: Optional[str] = None

@dataclass
class EnrichedContact:
    """Normalized enriched contact data structure"""
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    company_name: Optional[str] = None
    company_domain: Optional[str] = None
    linkedin_url: Optional[str] = None
    phone_numbers: List[str] = None
    location: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    confidence_score: Optional[float] = None
    last_updated: datetime = None

@dataclass
class EnrichedCompany:
    """Normalized enriched company data structure"""
    id: str
    name: str
    domain: str
    description: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    revenue_range: Optional[str] = None
    headquarters: Optional[str] = None
    founded_year: Optional[int] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    facebook_url: Optional[str] = None
    confidence_score: Optional[float] = None
    last_updated: datetime = None

class ApolloAPIError(Exception):
    """Custom exception for Apollo.io API errors"""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}

class ApolloConnector:
    """
    Production-ready Apollo.io connector with comprehensive error handling,
    rate limiting, and normalized data structures.
    """

    def __init__(self, credentials: ApolloCredentials):
        """
        Initialize the Apollo.io connector

        Args:
            credentials: Apollo.io API credentials
        """
        self.credentials = credentials
        self._session = self._create_session()

        # API endpoints
        self.endpoints = {
            'people_enrich': f"{credentials.base_url}/people/match",
            'people_search': f"{credentials.base_url}/people/search",
            'organizations_enrich': f"{credentials.base_url}/organizations/match",
            'organizations_search': f"{credentials.base_url}/organizations/search"
        }

        # Rate limiting: Apollo.io allows 100 requests per minute
        self.rate_limit_remaining = 100
        self.rate_limit_reset_time = datetime.now() + timedelta(minutes=1)

        logger.info("Apollo.io connector initialized successfully")

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
            'Content-Type': 'application/json',
            'X-API-Key': self.credentials.api_key
        })

        return session

    def _handle_rate_limiting(self):
        """Handle rate limiting by checking and waiting if necessary"""
        now = datetime.now()

        if now >= self.rate_limit_reset_time:
            self.rate_limit_remaining = 100
            self.rate_limit_reset_time = now + timedelta(minutes=1)

        if self.rate_limit_remaining <= 0:
            wait_time = (self.rate_limit_reset_time - now).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit exceeded, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.rate_limit_remaining = 100
                self.rate_limit_reset_time = datetime.now() + timedelta(minutes=1)

    def _make_api_request(self, method: str, url: str, **kwargs) -> dict:
        """
        Make an authenticated API request with error handling and rate limiting

        Args:
            method: HTTP method
            url: API endpoint URL
            **kwargs: Additional request parameters

        Returns:
            API response data

        Raises:
            ApolloAPIError: For API errors
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Handle rate limiting
                self._handle_rate_limiting()

                logger.debug(f"Making {method} request to {url}")

                response = self._session.request(method, url, **kwargs)

                # Update rate limit tracking
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', self.rate_limit_remaining - 1))

                # Handle different response codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise ApolloAPIError(
                        "Invalid API key",
                        response.status_code,
                        response.json() if response.content else None
                    )
                elif response.status_code == 403:
                    raise ApolloAPIError(
                        "API access forbidden - check permissions",
                        response.status_code,
                        response.json() if response.content else None
                    )
                elif response.status_code == 404:
                    raise ApolloAPIError(
                        "Resource not found",
                        response.status_code,
                        response.json() if response.content else None
                    )
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                else:
                    raise ApolloAPIError(
                        f"API request failed: {response.status_code}",
                        response.status_code,
                        response.json() if response.content else None
                    )

            except requests.RequestException as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise ApolloAPIError(f"Network error after {max_retries} retries: {str(e)}")
                logger.warning(f"Network error, retrying ({retry_count}/{max_retries}): {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff

        raise ApolloAPIError("Max retries exceeded")

    def enrich_contact(self, request: EnrichmentRequest) -> Optional[EnrichedContact]:
        """
        Enrich contact information using Apollo.io

        Args:
            request: Enrichment request with contact details

        Returns:
            Enriched contact data or None if not found
        """
        try:
            logger.info(f"Enriching contact: {request.email or request.first_name + ' ' + (request.last_name or '')}")

            # Prepare request data
            data = {}
            if request.email:
                data['email'] = request.email
            if request.first_name:
                data['first_name'] = request.first_name
            if request.last_name:
                data['last_name'] = request.last_name
            if request.company_name:
                data['organization_name'] = request.company_name
            if request.linkedin_url:
                data['linkedin_url'] = request.linkedin_url

            if not data:
                raise ValueError("At least one identifier (email, name, etc.) must be provided")

            response_data = self._make_api_request('POST', self.endpoints['people_enrich'], json=data)

            if not response_data.get('person'):
                logger.info("No contact data found for enrichment request")
                return None

            person = response_data['person']

            # Extract phone numbers
            phone_numbers = []
            if person.get('phone_numbers'):
                phone_numbers = [phone['number'] for phone in person['phone_numbers'] if phone.get('number')]

            enriched_contact = EnrichedContact(
                id=person.get('id', ''),
                email=person.get('email', request.email or ''),
                first_name=person.get('first_name'),
                last_name=person.get('last_name'),
                title=person.get('title'),
                company_name=person.get('organization', {}).get('name'),
                company_domain=person.get('organization', {}).get('primary_domain'),
                linkedin_url=person.get('linkedin_url'),
                phone_numbers=phone_numbers,
                location=person.get('city') + ', ' + person.get('state', '') + ' ' + person.get('country', '') if person.get('city') else None,
                industry=person.get('organization', {}).get('industry'),
                company_size=person.get('organization', {}).get('estimated_num_employees'),
                confidence_score=response_data.get('confidence_score'),
                last_updated=datetime.now()
            )

            logger.info(f"Successfully enriched contact: {enriched_contact.email}")
            return enriched_contact

        except Exception as e:
            logger.error(f"Failed to enrich contact: {e}")
            raise

    def enrich_company(self, request: EnrichmentRequest) -> Optional[EnrichedCompany]:
        """
        Enrich company information using Apollo.io

        Args:
            request: Enrichment request with company details

        Returns:
            Enriched company data or None if not found
        """
        try:
            logger.info(f"Enriching company: {request.domain or request.company_name}")

            # Prepare request data
            data = {}
            if request.domain:
                data['domain'] = request.domain
            if request.company_name:
                data['name'] = request.company_name

            if not data:
                raise ValueError("At least one identifier (domain or company name) must be provided")

            response_data = self._make_api_request('POST', self.endpoints['organizations_enrich'], json=data)

            if not response_data.get('organization'):
                logger.info("No company data found for enrichment request")
                return None

            org = response_data['organization']

            enriched_company = EnrichedCompany(
                id=org.get('id', ''),
                name=org.get('name', request.company_name or ''),
                domain=org.get('primary_domain', request.domain or ''),
                description=org.get('description'),
                industry=org.get('industry'),
                company_size=org.get('estimated_num_employees'),
                revenue_range=org.get('revenue_range'),
                headquarters=org.get('headquarters'),
                founded_year=org.get('founded_year'),
                linkedin_url=org.get('linkedin_url'),
                twitter_url=org.get('twitter_url'),
                facebook_url=org.get('facebook_url'),
                confidence_score=response_data.get('confidence_score'),
                last_updated=datetime.now()
            )

            logger.info(f"Successfully enriched company: {enriched_company.name}")
            return enriched_company

        except Exception as e:
            logger.error(f"Failed to enrich company: {e}")
            raise

    def search_contacts(self, query: str, page: int = 1, per_page: int = 10) -> List[EnrichedContact]:
        """
        Search for contacts using Apollo.io

        Args:
            query: Search query
            page: Page number
            per_page: Results per page

        Returns:
            List of enriched contact data
        """
        try:
            logger.info(f"Searching contacts: {query}")

            params = {
                'q': query,
                'page': page,
                'per_page': min(per_page, 100)  # API limit
            }

            response_data = self._make_api_request('GET', self.endpoints['people_search'], params=params)

            contacts = []
            for person in response_data.get('people', []):
                phone_numbers = []
                if person.get('phone_numbers'):
                    phone_numbers = [phone['number'] for phone in person['phone_numbers'] if phone.get('number')]

                contact = EnrichedContact(
                    id=person.get('id', ''),
                    email=person.get('email', ''),
                    first_name=person.get('first_name'),
                    last_name=person.get('last_name'),
                    title=person.get('title'),
                    company_name=person.get('organization', {}).get('name'),
                    company_domain=person.get('organization', {}).get('primary_domain'),
                    linkedin_url=person.get('linkedin_url'),
                    phone_numbers=phone_numbers,
                    location=person.get('city') + ', ' + person.get('state', '') + ' ' + person.get('country', '') if person.get('city') else None,
                    industry=person.get('organization', {}).get('industry'),
                    company_size=person.get('organization', {}).get('estimated_num_employees'),
                    confidence_score=person.get('confidence_score'),
                    last_updated=datetime.now()
                )
                contacts.append(contact)

            logger.info(f"Successfully found {len(contacts)} contacts")
            return contacts

        except Exception as e:
            logger.error(f"Failed to search contacts: {e}")
            raise

    def search_companies(self, query: str, page: int = 1, per_page: int = 10) -> List[EnrichedCompany]:
        """
        Search for companies using Apollo.io

        Args:
            query: Search query
            page: Page number
            per_page: Results per page

        Returns:
            List of enriched company data
        """
        try:
            logger.info(f"Searching companies: {query}")

            params = {
                'q': query,
                'page': page,
                'per_page': min(per_page, 100)  # API limit
            }

            response_data = self._make_api_request('GET', self.endpoints['organizations_search'], params=params)

            companies = []
            for org in response_data.get('organizations', []):
                company = EnrichedCompany(
                    id=org.get('id', ''),
                    name=org.get('name', ''),
                    domain=org.get('primary_domain', ''),
                    description=org.get('description'),
                    industry=org.get('industry'),
                    company_size=org.get('estimated_num_employees'),
                    revenue_range=org.get('revenue_range'),
                    headquarters=org.get('headquarters'),
                    founded_year=org.get('founded_year'),
                    linkedin_url=org.get('linkedin_url'),
                    twitter_url=org.get('twitter_url'),
                    facebook_url=org.get('facebook_url'),
                    confidence_score=org.get('confidence_score'),
                    last_updated=datetime.now()
                )
                companies.append(company)

            logger.info(f"Successfully found {len(companies)} companies")
            return companies

        except Exception as e:
            logger.error(f"Failed to search companies: {e}")
            raise

    def get_connection_status(self) -> dict:
        """
        Check the connection status to Apollo.io APIs

        Returns:
            Status information dictionary
        """
        try:
            # Test with a simple search to verify connectivity
            response_data = self._make_api_request('GET', self.endpoints['people_search'], params={'q': 'test', 'per_page': 1})

            return {
                'status': 'connected',
                'timestamp': datetime.now().isoformat(),
                'api_key_valid': True,
                'rate_limit_remaining': self.rate_limit_remaining,
                'message': 'Successfully connected to Apollo.io APIs'
            }

        except ApolloAPIError as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'api_key_valid': False,
                'error': str(e),
                'message': 'Failed to connect to Apollo.io APIs'
            }

        except Exception as e:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'api_key_valid': False,
                'error': str(e),
                'message': 'Unexpected error during connection test'
            }

# Factory function for easy initialization
def create_apollo_connector() -> ApolloConnector:
    """
    Create an Apollo.io connector instance using environment variables

    Returns:
        Configured ApolloConnector instance
    """
    api_key = os.environ.get('APOLLO_API_KEY', '')

    if not api_key:
        raise ValueError("Missing required Apollo.io API key in environment variables")

    credentials = ApolloCredentials(api_key=api_key)
    return ApolloConnector(credentials)
