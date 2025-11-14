# JPMorgan Financial APIs - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the JPMorgan Financial APIs to production environments.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ or CentOS 7+), Windows Server 2019+, or macOS 10.15+
- **CPU**: 2+ cores
- **RAM**: 4GB+ minimum, 8GB+ recommended
- **Storage**: 20GB+ free space
- **Network**: Stable internet connection

### Software Requirements
- **Docker**: 20.10+ with Docker Compose
- **Git**: 2.25+
- **Python**: 3.11+ (for local development)
- **PostgreSQL**: 15+ (optional, for production database)
- **Redis**: 7+ (optional, for caching)

### Security Requirements
- SSL/TLS certificates (Let's Encrypt or commercial)
- Firewall configuration
- Regular security updates
- Backup strategy

## Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd jpmorgan_financial_apis
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env.production
   # Edit .env.production with your production values
   nano .env.production
   ```

3. **Deploy**
   ```bash
   # Make deployment script executable
   chmod +x deploy_production.sh

   # Run deployment
   ./deploy_production.sh
   ```

### Option 2: Manual Docker Deployment

1. **Build and run**
   ```bash
   # Build production image
   docker build -t jpmorgan-apis:latest -f Dockerfile .

   # Run with docker-compose
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Option 3: Traditional Server Deployment

1. **Install dependencies**
   ```bash
   pip install -r requirements_new.txt
   pip install waitress
   ```

2. **Configure environment**
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secure-secret-key
   ```

3. **Run production server**
   ```bash
   python production_server.py
   ```

## Environment Configuration

### Production Environment Variables

Create a `.env.production` file with the following variables:

```bash
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-production-secret-key-change-this-in-production
FLASK_RUN_PORT=8000

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/jpmorgan_apis
# For SQLite (development only): DATABASE_URL=sqlite:///production.db

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password

# Authentication
TOKEN_CLIENT_ID=your-oauth-client-id
TOKEN_CLIENT_SECRET=your-oauth-client-secret
TOKEN_URL=https://your-oauth-provider.com/oauth/token
TOKEN_SCOPE=read write

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security
SESSION_COOKIE_SECURE=True
SESSION_COOKIE_HTTPONLY=True
SESSION_COOKIE_SAMESITE=Lax

# Rate Limiting
RATE_LIMIT_DEFAULT=200 per day
RATE_LIMIT_HEALTH=10 per minute
RATE_LIMIT_AUTH=5 per minute
RATE_LIMIT_TELEMETRY=5 per minute
RATE_LIMIT_ML=2 per minute

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_ADMIN_PASSWORD=secure-admin-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@jpmorgan.com
SMTP_PASSWORD=app-specific-password

# Feature Flags
ENABLE_ML_ANOMALY_DETECTION=true
ENABLE_TELEMETRY_PROCESSING=true
ENABLE_BUSINESS_ASSET_MANAGEMENT=true
ENABLE_DATA_CONVERSION=true
ENABLE_CLOUD_STORAGE=true

# Logging
LOG_LEVEL=INFO

# Backup
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
```

## Database Setup

### PostgreSQL (Recommended for Production)

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql postgresql-contrib

   # CentOS/RHEL
   sudo yum install postgresql-server postgresql-contrib
   sudo postgresql-setup initdb
   ```

2. **Create database and user**
   ```bash
   sudo -u postgres psql
   CREATE DATABASE jpmorgan_apis;
   CREATE USER jpmorgan_app WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE jpmorgan_apis TO jpmorgan_app;
   \q
   ```

3. **Run database migrations**
   ```bash
   # The init.sql script will be run automatically by docker-compose
   # For manual setup:
   psql -U jpmorgan_app -d jpmorgan_apis -f scripts/init.sql
   ```

### SQLite (Development Only)

SQLite is configured by default for development. For production, use PostgreSQL.

## SSL/TLS Configuration

### Using Let's Encrypt (Recommended)

1. **Install Certbot**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   ```

2. **Obtain certificate**
   ```bash
   sudo certbot --nginx -d your-domain.com
   ```

3. **Auto-renewal**
   ```bash
   sudo crontab -e
   # Add: 0 12 * * * /usr/bin/certbot renew --quiet
   ```

### Manual Certificate Installation

1. **Place certificates**
   ```bash
   sudo mkdir -p /etc/nginx/ssl
   sudo cp your-domain.crt /etc/nginx/ssl/cert.pem
   sudo cp your-domain.key /etc/nginx/ssl/key.pem
   ```

2. **Update nginx configuration**
   ```nginx
   server {
       listen 443 ssl http2;
       server_name your-domain.com;

       ssl_certificate /etc/nginx/ssl/cert.pem;
       ssl_certificate_key /etc/nginx/ssl/key.pem;
       # ... rest of SSL configuration
   }
   ```

## Monitoring Setup

### Prometheus & Grafana

The docker-compose.prod.yml includes Prometheus, Grafana, and AlertManager.

1. **Access Grafana**
   - URL: http://localhost:3000
   - Default credentials: admin / admin (change immediately)

2. **Import dashboard**
   ```bash
   # Dashboard JSON is available at grafana_dashboard.json
   # Import through Grafana UI
   ```

3. **Configure alerts**
   ```bash
   # AlertManager configuration in alertmanager.yml
   # Email alerts configured via SMTP settings
   ```

### Health Checks

- **API Health**: `GET /health`
- **Container Health**: `docker ps`
- **Logs**: `docker-compose logs -f`

## Backup Strategy

### Automated Backups

1. **Database backups**
   ```bash
   # Configured in docker-compose.prod.yml
   # Daily backups at 2 AM
   ```

2. **Application backups**
   ```bash
   # Deployment script creates backups before updates
   ./deploy_production.sh  # Creates timestamped backup
   ```

3. **Manual backup**
   ```bash
   # Backup database
   pg_dump -U jpmorgan_app jpmorgan_apis > backup_$(date +%Y%m%d).sql

   # Backup application data
   tar -czf app_backup_$(date +%Y%m%d).tar.gz /opt/jpmorgan-financial-apis
   ```

## Scaling

### Horizontal Scaling

1. **Multiple application instances**
   ```yaml
   # In docker-compose.prod.yml
   services:
     jpmorgan-apis:
       scale: 3  # Run 3 instances
   ```

2. **Load balancer**
   ```yaml
   # Add nginx load balancer
   services:
     nginx-lb:
       image: nginx:alpine
       ports:
         - "80:80"
       volumes:
         - ./nginx/lb.conf:/etc/nginx/nginx.conf
   ```

### Vertical Scaling

1. **Increase resources**
   ```yaml
   services:
     jpmorgan-apis:
       deploy:
         resources:
           limits:
             cpus: '2.0'
             memory: 4G
           reservations:
             cpus: '1.0'
             memory: 2G
   ```

## Security Best Practices

### Network Security

1. **Firewall configuration**
   ```bash
   # UFW example
   sudo ufw allow 80
   sudo ufw allow 443
   sudo ufw allow 22
   sudo ufw --force enable
   ```

2. **Fail2Ban**
   ```bash
   sudo apt install fail2ban
   sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
   sudo systemctl enable fail2ban
   sudo systemctl start fail2ban
   ```

### Application Security

1. **Regular updates**
   ```bash
   # Update dependencies
   pip install --upgrade -r requirements_new.txt

   # Update base images
   docker-compose pull
   docker-compose up -d
   ```

2. **Security scanning**
   ```bash
   # Scan for vulnerabilities
   docker scan jpmorgan-apis:latest
   ```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check what's using port 8000
   sudo lsof -i :8000
   sudo netstat -tulpn | grep :8000
   ```

2. **Container fails to start**
   ```bash
   # Check logs
   docker-compose logs jpmorgan-apis

   # Check container status
   docker-compose ps
   ```

3. **Database connection issues**
   ```bash
   # Test database connection
   psql -U jpmorgan_app -d jpmorgan_apis -h localhost

   # Check PostgreSQL logs
   sudo tail -f /var/log/postgresql/postgresql-*.log
   ```

4. **Memory issues**
   ```bash
   # Monitor memory usage
   docker stats

   # Check system memory
   free -h
   ```

### Logs and Debugging

1. **Application logs**
   ```bash
   # View application logs
   docker-compose logs -f jpmorgan-apis

   # View all logs
   docker-compose logs -f
   ```

2. **System logs**
   ```bash
   # System logs
   sudo journalctl -u docker -f

   # Nginx logs
   sudo tail -f /var/log/nginx/error.log
   ```

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor health checks
   - Review error logs
   - Check disk space

2. **Weekly**
   - Update dependencies
   - Review security alerts
   - Test backup restoration

3. **Monthly**
   - Security patches
   - Performance optimization
   - Log rotation

### Updates

1. **Rolling updates**
   ```bash
   # Update without downtime
   docker-compose pull
   docker-compose up -d --scale jpmorgan-apis=2
   docker-compose up -d --scale jpmorgan-apis=1
   ```

2. **Blue-green deployment**
   ```bash
   # Deploy new version alongside old
   docker-compose -f docker-compose.green.yml up -d
   # Test green environment
   # Switch traffic to green
   docker-compose -f docker-compose.blue.yml down
   ```

## Support

For production support and issues:

1. **Check logs first**
2. **Review monitoring dashboards**
3. **Test in staging environment**
4. **Contact development team**

## API Documentation

Once deployed, access the API documentation at:
- **Swagger UI**: `http://your-domain.com/swagger/`
- **Health Check**: `http://your-domain.com/health`
- **API Root**: `http://your-domain.com/`

## License

This project is proprietary to JPMorgan Chase & Co.
