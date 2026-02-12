# JPMorgan Financial APIs - Enterprise-Grade API System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-blue.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> Enterprise-grade API system for processing JPMorgan financial data with ML anomaly detection, comprehensive audit logging, and production monitoring.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Organization Administration Overview](#organization-administration-overview)
- [Migration](#-migration)
- [Ask Gordon Integration](#-ask-gordon-integration)
- [Monitoring](#-monitoring)
- [Security](#-security)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Features

### Core Functionality
- **Real-time Financial Data Processing** - Process JPMorgan financial metrics and stock data
- **Machine Learning Anomaly Detection** - Automated detection of unusual patterns
- **Business & Asset Management** - CRUD operations for business entities and assets
- **Revenue Tracking** - Comprehensive transaction and revenue management
- **Audit Logging** - Tamper-proof audit trails with hash chains
- **Apollo.io Data Enrichment** - Contact and company data enrichment

### Enterprise Features
- **Kong API Gateway** - Rate limiting, authentication, and request routing
- **Prometheus Monitoring** - Comprehensive metrics collection
- **Grafana Dashboards** - Real-time visualization and alerting
- **PostgreSQL Database** - Robust data persistence with connection pooling
- **Redis Caching** - High-performance caching layer
- **WebSocket Support** - Real-time communication
- **Comprehensive Security** - JWT authentication, CORS, rate limiting

### Production Ready
- **Docker Containerization** - Complete containerized deployment
- **Health Checks** - Automated service monitoring
- **Load Balancing** - Built-in support for horizontal scaling
- **Backup & Recovery** - Automated database backups
- **CI/CD Ready** - GitLab CI/CD pipeline configuration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Kong Gateway  │────│  Flask API      │────│  PostgreSQL     │
│                 │    │                 │    │  Database       │
│ • Rate Limiting │    │ • Business Logic│    │                 │
│ • Authentication│    │ • ML Models     │    │ • Audit Logs    │
│ • Request Trans │    │ • Data Processing│   │ • Transactions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Monitoring    │
                    │                 │
                    │ • Prometheus    │
                    │ • Grafana       │
                    │ • Node Exporter │
                    └─────────────────┘
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.8+ (for local development)
- 4GB RAM minimum, 8GB recommended

### 1. Clone and Setup
```bash
git clone <repository-url>
cd jpmorgan_financial_apis
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

### 3. Launch with Docker Compose
```bash
# Start all services
docker-compose -f docker-compose.kong.yml up -d

# Check service status
docker-compose -f docker-compose.kong.yml ps
```

### 4. Verify Installation
```bash
# Health check
curl http://localhost:5000/health

# Kong Gateway
curl http://localhost:8000/health

# Grafana Dashboard
open http://localhost:3000  # admin/admin
```

## 📚 API Documentation

### Core Endpoints

#### Health & Status
- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics

#### Authentication
- `POST /user/register` - User registration
- `POST /user/login` - User authentication
- `GET /user/profile` - User profile (JWT required)

#### Telemetry Processing
- `POST /telemetry` - Process single telemetry event
- `POST /telemetry/batch` - Process batch telemetry events
- `GET /telemetry/metrics` - Get telemetry metrics
- `POST /ml/anomalies` - Detect anomalies with ML

#### Business Management
- `GET /businesses` - List businesses
- `POST /businesses` - Create business
- `GET /businesses/{id}` - Get business details
- `PUT /businesses/{id}` - Update business
- `DELETE /businesses/{id}` - Delete business

#### Asset Management
- `GET /assets` - List assets
- `POST /assets` - Create asset
- `GET /assets/{id}` - Get asset details
- `PUT /assets/{id}` - Update asset
- `DELETE /assets/{id}` - Delete asset

#### Revenue Tracking
- `POST /revenue/transactions` - Create revenue transaction
- `GET /revenue/transactions` - List user transactions
- `GET /revenue/metrics` - Get revenue metrics

#### JPMorgan Data
- `GET /api/jpmorgan-data` - Get financial metrics and stock data

#### Audit Logging
- `GET /audit/logs` - Query audit logs
- `GET /audit/summary` - Get audit statistics
- `POST /audit/verify-integrity` - Verify audit log integrity

#### Apollo.io Enrichment
- `POST /enrichment/contact` - Enrich contact information
- `POST /enrichment/company` - Enrich company information
- `GET /enrichment/search/contacts` - Search contacts
- `GET /enrichment/search/companies` - Search companies

#### AI-Powered Financial Analysis
- `POST /ai/financial-context` - Analyze financial context from transaction data
- `GET /ai/financial-context/<analysis_id>` - Retrieve stored financial context analysis
- `POST /ai/verify-identity` - Verify user identity with document/liveness checks
- `GET /ai/verify-identity/<verification_id>` - Retrieve identity verification results
- `POST /ai/know-your-agent` - Implement Know Your Agent (KYA) workflow
- `POST /ai/agentic-commerce/pay-by-bank` - Enable pay-by-bank functionality
- `POST /ai/agentic-commerce/fund-wallet` - Fund digital wallet from bank account
- `GET /ai/agentic-commerce/transactions/<transaction_id>` - Retrieve agentic commerce transaction
- `POST /ai/query` - Process natural language queries about financial data
- `POST /ai/risk-assess` - Assess risk for financial transactions
- `GET /ai/status` - Get AI service status and health metrics
- `GET /ai/dashboard` - Get AI dashboard overview with usage statistics

#### Machine Learning Financial Analysis
- `POST /ml/anomalies` - Detect anomalies in financial data
- `POST /ml/train` - Train ML models with financial data
- `GET /ml/models` - List available ML models
- `GET /ml/models/<model_id>` - Get details of specific ML model
- `POST /ml/predict/<model_id>` - Make predictions using trained model
- `POST /ml/models/<model_id>/evaluate` - Evaluate model performance
- `GET /ml/models/<model_id>/features` - Get feature importance for model
- `GET /ml/monitoring` - Get ML system monitoring data
- `GET /ml/dashboard` - Get ML dashboard overview
- `POST /ml/financial-context` - Analyze financial context from transaction data
- `POST /ml/transaction-patterns` - Analyze transaction patterns for behavioral insights
- `GET /ml/spending-insights` - Get personalized spending insights and recommendations
- `POST /ml/cash-flow-analysis` - Analyze cash flow patterns and projections

#### Financial Data Management
- `POST /data/financial/transactions` - Create financial transaction records
- `GET /data/financial/transactions` - Get financial transactions with filtering
- `GET /data/financial/accounts` - Get account balance information
- `GET /data/financial/user-data/<user_id>` - Get user-permissioned financial data

### Authentication
Most endpoints require JWT authentication. Include the token in headers:
```
Authorization: Bearer <your-jwt-token>
```

## 🚢 Deployment

### Production Deployment

#### 1. Environment Setup
```bash
# Set production environment variables
export FLASK_ENV=production
export SECRET_KEY=$(openssl rand -hex 32)
export JWT_SECRET_KEY=$(openssl rand -hex 32)
```

#### 2. Database Initialization
```bash
# Initialize database
python init_database.py
```

#### 3. Docker Deployment
```bash
# Build and deploy
docker-compose -f docker-compose.kong.yml up --build -d

# Scale services if needed
docker-compose -f docker-compose.kong.yml up -d --scale flask-api=3
```

#### 4. SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Update Kong configuration for HTTPS
# Edit kong.yml to include certificate paths
```

### Kubernetes Deployment
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

## 🏢 Organization Administration Overview

A Docker organization is a collection of teams and repositories with centralized
management. It helps administrators group members and assign access in a
streamlined, scalable way.

### Organization structure

The following diagram shows how organizations relate to teams and members.

![Diagram showing how teams and members relate within a Docker organization](/admin/images/org-structure.webp)

### Organization members

Organization owners have full administrator access to manage members, roles,
and teams across the organization.

An organization includes members and optional teams. Teams help group members
and simplify permission management.

### Create and manage your organization

Learn how to create and manage your organization in the following sections.

## 🔄 Migration

This section provides guidance for migrating your applications to Docker
Hardened Images (DHI). Migrating to DHI enhances the security posture of your
containerized applications by leveraging hardened base images with built-in
security features.

### Migration paths

Choose the migration approach that best fits your needs:

### Resources

## 🤖 Ask Gordon Integration

This project leverages **Ask Gordon**, Docker's AI assistant, to streamline Docker workflows and improve container management. Ask Gordon provides intelligent assistance for Docker-related tasks, making it easier to work with complex multi-service applications like this JPMorgan Financial APIs system.

### What Ask Gordon Can Do for This Project

Ask Gordon can help with:

- **Dockerfile Optimization** - Analyze and improve Dockerfiles for better performance and security
- **Container Troubleshooting** - Debug container startup issues, network problems, and service dependencies
- **Docker Compose Optimization** - Optimize multi-service configurations and resource allocation
- **Security Scanning** - Identify vulnerabilities in container images and configurations
- **Performance Tuning** - Suggest improvements for container resource usage and build efficiency
- **Migration to DHI** - Help migrate Dockerfiles to use Docker Hardened Images for enhanced security

### Getting Started with Ask Gordon

1. **Enable Ask Gordon** in Docker Desktop:
   - Go to Settings → Beta features
   - Check "Enable Docker AI"
   - Accept the terms and enable the feature

2. **Access Ask Gordon**:
   - In Docker Desktop: Look for the **Ask Gordon** view
   - In CLI: Use the `docker ai` command
   - Look for ✨ icons throughout Docker Desktop for contextual help

### Practical Examples for This Project

#### Troubleshooting Container Issues
```bash
# Navigate to the project directory
cd jpmorgan_financial_apis

# Ask Gordon to help with container issues
docker ai "My PostgreSQL container is failing to start. Can you help troubleshoot?"

# Or use the interactive mode
docker ai
```

#### Optimizing Docker Compose Performance
```bash
# Ask Gordon to analyze your Docker Compose setup
docker ai "Analyze my docker-compose.kong.yml file and suggest performance improvements"
```

#### Improving Dockerfiles
```bash
# Have Gordon rate and improve your Dockerfiles
docker ai rate my Dockerfile

# Or get specific recommendations
docker ai "How can I optimize my Flask API Dockerfile for production?"
```

#### Migrating to Docker Hardened Images
```bash
# Start interactive mode for complex tasks
docker ai

# Then type: "Migrate my dockerfile to DHI"
```

### Best Practices

- **Always verify AI suggestions** - While Ask Gordon is helpful, always test changes in a development environment first
- **Use contextual help** - When you see the ✨ icon in Docker Desktop, it provides project-specific assistance
- **Combine with monitoring** - Use Ask Gordon alongside your Grafana/Prometheus setup for comprehensive container management
- **Security first** - Ask Gordon can help identify security issues, but always review changes carefully

### Integration with Project Workflow

Ask Gordon works seamlessly with your existing development and deployment process:

- **Development**: Get help setting up local development environments
- **Testing**: Troubleshoot integration test failures
- **Deployment**: Optimize production Docker Compose configurations
- **Monitoring**: Analyze container performance and resource usage
- **Security**: Scan for vulnerabilities and apply hardening techniques

For complete Ask Gordon documentation, see the [Ask Gordon Guide](../ask-gordon.md).

## 📊 Monitoring

### Grafana Dashboards
- **Executive Dashboard** - High-level business metrics
- **API Performance** - Response times and throughput
- **Security Monitoring** - Failed logins and suspicious activity
- **System Health** - CPU, memory, and disk usage

### Prometheus Metrics
- **HTTP Request Metrics** - Request count, latency, error rates
- **Business Metrics** - Transaction volumes, revenue tracking
- **Security Metrics** - Authentication attempts, audit events
- **System Metrics** - Resource usage and health checks

### Alerting
- Response time > 500ms
- Error rate > 5%
- Failed login attempts > 5/min
- Database connection issues

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (USER, ADMIN)
- Password hashing with bcrypt
- Session management

### API Security
- Rate limiting (Kong Gateway)
- CORS protection
- Input validation and sanitization
- SQL injection prevention
- XSS protection

### Data Protection
- Audit logging with tamper-proof hash chains
- Encryption at rest and in transit
- GDPR compliance features
- Data retention policies

### Infrastructure Security
- Container security scanning
- Network segmentation
- Secret management with environment variables
- Regular security updates

## 🧪 Testing

### Unit Tests
```bash
# Run unit tests
python -m pytest tests/ -v
```

### Integration Tests
```bash
# Run comprehensive E2E tests
python comprehensive_e2e_test.py
```

### Performance Tests
```bash
# Run performance benchmarks
python performance_test.py
```

### Load Testing
```bash
# Run with Locust
locust -f locustfile.py --host=http://localhost:5000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
export FLASK_ENV=development
python app_final.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [API Docs](docs/)
- **Issues**: [GitHub Issues](issues/)
- **Discussions**: [GitHub Discussions](discussions/)

## 🙏 Acknowledgments

- JPMorgan Chase for financial data integration
- Kong for API gateway technology
- Prometheus and Grafana communities
- Flask and Python communities

---

**Made with ❤️ for enterprise-grade financial API solutions**
