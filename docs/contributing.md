# Contributing Guidelines - JPMorgan Financial APIs

## Overview

Thank you for your interest in contributing to the JPMorgan Financial APIs project! This document provides guidelines and best practices for contributing to this enterprise-grade financial platform.

## Getting Started

### Development Environment Setup

#### Prerequisites

- **Python**: 3.8 or higher
- **Docker**: 20.10+ with Docker Compose
- **Git**: 2.30+ with Git LFS
- **Kubernetes**: kubectl 1.24+ (for local development)
- **Helm**: 3.8+ (optional, for advanced deployments)

#### Local Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/jpmorgan-financial-apis.git
cd jpmorgan-financial-apis

# 2. Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Set up pre-commit hooks
pip install pre-commit
pre-commit install

# 5. Configure environment variables
cp .env.example .env
# Edit .env with your development credentials

# 6. Run database migrations
python scripts/postgresql_migration.py

# 7. Start development services
docker-compose up -d postgresql redis

# 8. Run the application
python app.py
```

#### Testing Your Setup

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/e2e/

# Run linting
flake8 src/
black --check src/
mypy src/
```

## Development Workflow

### Branching Strategy

We use a Git Flow-inspired branching strategy:

```
main (production-ready)
├── develop (integration branch)
│   ├── feature/FEATURE-123-user-authentication
│   ├── feature/FEATURE-124-api-rate-limiting
│   ├── bugfix/BUG-456-memory-leak-fix
│   └── hotfix/HOTFIX-789-critical-security-patch
```

#### Branch Naming Conventions

- **Features**: `feature/FEATURE-123-description`
- **Bug Fixes**: `bugfix/BUG-456-description`
- **Hotfixes**: `hotfix/HOTFIX-789-description`
- **Documentation**: `docs/DOCS-123-description`

### Commit Message Format

We follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

#### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

#### Examples

```bash
feat(auth): add OAuth2 client credentials flow

fix(api): resolve memory leak in account listing endpoint

docs(api): update authentication examples in API docs

test(auth): add unit tests for token validation

refactor(db): optimize query performance for large datasets
```

### Pull Request Process

#### Before Submitting

1. **Update Documentation**: Ensure all changes are documented
2. **Add Tests**: Write comprehensive tests for new features
3. **Run Quality Checks**: All checks must pass
4. **Update CHANGELOG**: Add entry for user-facing changes

#### PR Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security enhancement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing performed
- [ ] Load testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] CHANGELOG updated
- [ ] Security review completed
- [ ] Performance impact assessed

## Additional Notes
Any additional information or context.
```

#### Code Review Process

1. **Automated Checks**: CI/CD pipeline runs all quality checks
2. **Peer Review**: At least one maintainer reviews the code
3. **Security Review**: Security team reviews for vulnerabilities
4. **Merge**: Squash merge with descriptive commit message

## Code Quality Standards

### Python Style Guide

We follow PEP 8 with some additional rules:

```python
# Good: Descriptive variable names
user_account_balance = get_account_balance(user_id)

# Bad: Non-descriptive names
uab = get_ab(uid)

# Good: Type hints
def calculate_interest(principal: float, rate: float, time: int) -> float:
    return principal * rate * time

# Good: Docstrings
def get_user_accounts(user_id: str) -> List[Account]:
    """
    Retrieve all accounts for a given user.

    Args:
        user_id: Unique identifier for the user

    Returns:
        List of Account objects

    Raises:
        UserNotFoundError: If user doesn't exist
    """
    pass

# Good: Error handling
try:
    account = get_account(account_id)
except AccountNotFoundError:
    logger.warning(f"Account {account_id} not found")
    return None
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    raise APIError("Internal server error")
```

### Code Organization

```
src/
├── __init__.py
├── app.py                 # Main Flask application
├── config.py              # Configuration management
├── token_manager.py       # OAuth token handling
├── circuit_breaker.py     # Circuit breaker implementation
├── user_manager.py        # User management
├── account_service.py      # Account business logic
├── market_service.py       # Market data service
├── trading_service.py      # Trading operations
├── database/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy models
│   └── migrations/        # Database migrations
├── api/
│   ├── __init__.py
│   ├── accounts.py        # Account endpoints
│   ├── market.py          # Market data endpoints
│   └── trading.py         # Trading endpoints
├── utils/
│   ├── __init__.py
│   ├── validation.py      # Input validation
│   ├── caching.py         # Caching utilities
│   └── security.py        # Security utilities
└── tests/
    ├── __init__.py
    ├── conftest.py        # Test configuration
    ├── unit/
    ├── integration/
    └── e2e/
```

### Testing Standards

#### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestAccountService:
    def setup_method(self):
        self.service = AccountService()

    def test_get_account_success(self):
        # Arrange
        account_id = "12345"
        expected_account = Account(id=account_id, balance=1000.0)

        with patch.object(self.service.db, 'get_account') as mock_get:
            mock_get.return_value = expected_account

            # Act
            result = self.service.get_account(account_id)

            # Assert
            assert result == expected_account
            mock_get.assert_called_once_with(account_id)

    def test_get_account_not_found(self):
        # Arrange
        account_id = "nonexistent"

        with patch.object(self.service.db, 'get_account') as mock_get:
            mock_get.return_value = None

            # Act & Assert
            with pytest.raises(AccountNotFoundError):
                self.service.get_account(account_id)
```

#### Integration Tests

```python
import pytest
from tests.fixtures import app, client

class TestAccountAPI:
    def test_get_accounts_authenticated(self, client, auth_token):
        # Arrange
        headers = {'Authorization': f'Bearer {auth_token}'}

        # Act
        response = client.get('/api/v1/accounts', headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert 'accounts' in data
        assert isinstance(data['accounts'], list)

    def test_get_accounts_unauthorized(self, client):
        # Act
        response = client.get('/api/v1/accounts')

        # Assert
        assert response.status_code == 401
```

#### End-to-End Tests

```python
import pytest
from tests.e2e.fixtures import live_api_client

class TestAccountE2E:
    def test_full_account_workflow(self, live_api_client):
        # This would test against a real environment
        # with actual JPMorgan API calls

        # Create test account
        account_data = {
            'account_name': 'Test Account',
            'currency': 'USD'
        }

        response = live_api_client.post('/api/v1/accounts', json=account_data)
        assert response.status_code == 201

        account_id = response.get_json()['account_id']

        # Retrieve account
        response = live_api_client.get(f'/api/v1/accounts/{account_id}')
        assert response.status_code == 200

        # Update account
        update_data = {'account_name': 'Updated Test Account'}
        response = live_api_client.put(f'/api/v1/accounts/{account_id}', json=update_data)
        assert response.status_code == 200

        # Delete account
        response = live_api_client.delete(f'/api/v1/accounts/{account_id}')
        assert response.status_code == 204
```

### Security Requirements

#### Input Validation

```python
from pydantic import BaseModel, validator
from typing import Optional

class AccountRequest(BaseModel):
    account_name: str
    currency: str
    initial_balance: Optional[float] = 0.0

    @validator('account_name')
    def validate_account_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Account name cannot be empty')
        if len(v) > 100:
            raise ValueError('Account name too long')
        return v.strip()

    @validator('currency')
    def validate_currency(cls, v):
        valid_currencies = ['USD', 'EUR', 'GBP', 'JPY']
        if v.upper() not in valid_currencies:
            raise ValueError(f'Invalid currency: {v}')
        return v.upper()

    @validator('initial_balance')
    def validate_balance(cls, v):
        if v < 0:
            raise ValueError('Initial balance cannot be negative')
        if v > 10000000:  # 10 million limit
            raise ValueError('Initial balance exceeds maximum allowed')
        return v
```

#### Secure Coding Practices

```python
# Good: Parameterized queries prevent SQL injection
def get_user_by_id(self, user_id: str):
    return self.db.session.execute(
        text("SELECT * FROM users WHERE id = :user_id"),
        {"user_id": user_id}
    ).fetchone()

# Bad: String concatenation vulnerable to SQL injection
# def get_user_by_id(self, user_id: str):
#     return self.db.session.execute(f"SELECT * FROM users WHERE id = '{user_id}'")

# Good: Input sanitization
import bleach

def sanitize_html_input(user_input: str) -> str:
    allowed_tags = ['p', 'br', 'strong', 'em']
    allowed_attrs = {}
    return bleach.clean(user_input, tags=allowed_tags, attributes=allowed_attrs)

# Good: Secure random generation
import secrets

def generate_secure_token(length: int = 32) -> str:
    return secrets.token_urlsafe(length)

# Bad: Using random instead of secrets
# import random
# def generate_insecure_token(length: int = 32) -> str:
#     return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
```

## Documentation Standards

### API Documentation

```python
from flask import Blueprint, jsonify, request
from flasgger import swag_from

accounts_bp = Blueprint('accounts', __name__)

@accounts_bp.route('/api/v1/accounts', methods=['GET'])
@swag_from({
    'tags': ['Accounts'],
    'summary': 'List user accounts',
    'description': 'Retrieve all accounts for the authenticated user',
    'parameters': [
        {
            'name': 'limit',
            'in': 'query',
            'type': 'integer',
            'default': 50,
            'description': 'Maximum number of accounts to return'
        },
        {
            'name': 'offset',
            'in': 'query',
            'type': 'integer',
            'default': 0,
            'description': 'Number of accounts to skip'
        }
    ],
    'responses': {
        '200': {
            'description': 'Successful response',
            'schema': {
                'type': 'object',
                'properties': {
                    'accounts': {
                        'type': 'array',
                        'items': {'$ref': '#/definitions/Account'}
                    },
                    'pagination': {'$ref': '#/definitions/Pagination'}
                }
            }
        },
        '401': {
            'description': 'Unauthorized'
        }
    },
    'security': [{'Bearer': []}]
})
def get_accounts():
    """List user accounts with pagination"""
    pass
```

### README Updates

When adding new features, update the main README.md:

```markdown
## New Feature: Advanced Analytics

The API now supports advanced portfolio analytics including:

- Risk metrics (VaR, Sharpe ratio, beta)
- Performance attribution
- Benchmarking against market indices
- Custom date ranges and frequencies

### Usage Example

```python
analytics = api.get_portfolio_analytics(
    account_id='12345',
    start_date='2024-01-01',
    end_date='2024-12-31',
    metrics=['sharpe_ratio', 'max_drawdown', 'beta']
)
```
```

## Performance Considerations

### Profiling and Optimization

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        result = func(*args, **kwargs)

        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 time-consuming functions

        return result
    return wrapper

# Usage
@profile_function
def expensive_operation():
    # Your code here
    pass
```

### Memory Optimization

```python
# Use generators for large datasets
def get_large_account_list():
    """Generator that yields accounts one by one"""
    query = Account.query
    for account in query.yield_per(100):  # Process in chunks
        yield account

# Instead of loading everything into memory
# accounts = Account.query.all()  # Bad for large datasets

# Use streaming for large responses
from flask import Response, stream_with_context

@app.route('/api/accounts/export')
def export_accounts():
    @stream_with_context
    def generate():
        yield 'account_id,balance,currency\n'
        for account in get_large_account_list():
            yield f"{account.id},{account.balance},{account.currency}\n"

    return Response(generate(), mimetype='text/csv')
```

## Security Review Process

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Security Scanning

```bash
# Run security scans before committing
pip install safety
safety check

# Scan for secrets
pip install detect-secrets
detect-secrets scan

# Run bandit security linter
pip install bandit
bandit -r src/
```

## Release Process

### Version Management

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Security review completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Migration scripts tested
- [ ] Rollback plan documented
- [ ] Deployment verified in staging

### Release Commands

```bash
# Create release branch
git checkout -b release/v2.1.0 develop

# Update version
echo "2.1.0" > VERSION

# Update CHANGELOG.md
# ... add release notes ...

# Commit changes
git add VERSION CHANGELOG.md
git commit -m "chore: release v2.1.0"

# Merge to main
git checkout main
git merge release/v2.1.0

# Tag release
git tag -a v2.1.0 -m "Release v2.1.0"

# Push
git push origin main --tags
```

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Slack**: Real-time communication (#contributors channel)
- **Documentation**: Comprehensive guides and API reference

### Support Levels

1. **Community Support**: GitHub issues and discussions
2. **Priority Support**: Security issues and critical bugs
3. **Enterprise Support**: Custom development and consulting

### Escalation Process

1. **Try Documentation**: Check docs and troubleshooting guides
2. **Search Issues**: Look for existing similar issues
3. **Create Issue**: Provide detailed reproduction steps
4. **Engage Maintainers**: Tag appropriate maintainers
5. **Escalate**: Contact enterprise support if needed

## Recognition

### Contributor Recognition

Contributors are recognized through:

- **GitHub Contributors**: Listed in repository contributors
- **CHANGELOG**: Mentioned in release notes
- **Hall of Fame**: Featured contributors page
- **Events**: Invited to contributor events

### Rewards Program

- **Bug Bounties**: Rewards for security-related fixes
- **Feature Grants**: Funding for significant feature development
- **Hackathons**: Regular coding challenges with prizes

Thank you for contributing to the JPMorgan Financial APIs project! Your contributions help make financial technology more accessible and reliable for everyone.

---

**Last Updated**: November 2024
**Version**: 1.0.0
