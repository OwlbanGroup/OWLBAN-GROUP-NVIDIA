# JPMorgan Financial APIs - Live Production Deployment Guide

## 🚀 Quick Start - Deploy to Production

This guide will help you deploy the JPMorgan Financial APIs to a live production environment.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04 LTS or later (recommended)
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: 4+ cores
- **Storage**: 50GB+ SSD
- **Network**: Static IP address with open ports 80, 443

### Software Requirements
- Docker 20.10+ and Docker Compose 2.0+
- OR
- Python 3.9+, PostgreSQL 14+, Redis 7+, NGINX

---

## 🎯 Deployment Options

### Option 1: Docker Compose (Recommended)

**Fastest and easiest deployment method**

#### Step 1: Prepare the Environment

```bash
# Clone or copy the project
cd /opt
git clone <your-repo> jpmorgan-financial-apis
cd jpmorgan-financial-apis

# Create necessary directories
mkdir -p logs backups nginx/ssl models
```

#### Step 2: Configure Environment

```bash
# Copy and edit production environment file
cp .env.production.example .env.production

# IMPORTANT: Update these values in .env.production
nano .env.production
```

**Critical settings to update:**
- `SECRET_KEY` - Generate a secure random key
- `JWT_SECRET_KEY` - Generate a secure random key
- `DATABASE_URL` - Update password
- `AWS_ACCESS_KEY_ID` - Your AWS credentials (if using S3)
- `GITHUB_TOKEN` - Your GitHub token (if using GitHub integration)

#### Step 3: Generate SSL Certificates

```bash
# Option A: Self-signed (for testing)
python3 scripts/setup_https.py --action generate-self-signed \
    --domain your-domain.com \
    --cert-dir nginx/ssl

# Option B: Let's Encrypt (for production)
sudo apt-get install certbot
sudo certbot certonly --standalone -d your-domain.com
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/server.crt
cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/server.key
```

#### Step 4: Deploy with Docker Compose

```bash
# Build and start all services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f app
```

#### Step 5: Verify Deployment

```bash
# Check health
curl -k https://localhost/health

# Run validation
docker-compose -f docker-compose.production.yml exec app \
    python scripts/prod_validation.py --url https://localhost

# Run compliance check
docker-compose -f docker-compose.production.yml exec app \
    python scripts/compliance_check.py --url https://localhost
```

---

### Option 2: Native Installation (Advanced)

**For maximum performance and control**

#### Step 1: Run Deployment Script

```bash
# Make script executable
chmod +x deploy_live_production.sh

# Run as root
sudo ./deploy_live_production.sh
```

The script will automatically:
- ✅ Install all dependencies
- ✅ Set up PostgreSQL and Redis
- ✅ Deploy the application
- ✅ Configure NGINX with SSL
- ✅ Set up monitoring (Prometheus, Grafana)
- ✅ Configure firewall
- ✅ Set up log rotation
- ✅ Run health checks

#### Step 2: Verify Installation

```bash
# Check application status
sudo supervisorctl status jpmorgan-api

# Check logs
tail -f /var/log/jpmorgan/gunicorn.out.log

# Test API
curl -k https://localhost/health
```

---

## 🔧 Post-Deployment Configuration

### 1. Update DNS Records

Point your domain to the server's IP address:

```
A Record: api.yourdomain.com -> YOUR_SERVER_IP
```

### 2. Configure Firewall

```bash
# Using UFW (Ubuntu)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

### 3. Set Up Monitoring

#### Access Grafana Dashboard
- URL: `https://your-domain.com:3000`
- Username: `admin`
- Password: `SecureGrafanaP@ss2024` (change this!)

#### Access Prometheus
- URL: `https://your-domain.com:9090`

### 4. Configure Backups

```bash
# Set up automated backups (cron job)
sudo crontab -e

# Add this line for daily backups at 2 AM
0 2 * * * /opt/jpmorgan-financial-apis/scripts/backup.sh
```

### 5. Set Up SSL Certificate Auto-Renewal

```bash
# For Let's Encrypt certificates
sudo crontab -e

# Add this line
0 0 1 * * certbot renew --quiet && systemctl reload nginx
```

---

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# Application health
curl https://your-domain.com/health

# Database connection
docker-compose exec postgresql pg_isready

# Redis connection
docker-compose exec redis redis-cli ping
```

### View Logs

```bash
# Docker Compose
docker-compose -f docker-compose.production.yml logs -f app

# Native Installation
tail -f /var/log/jpmorgan/gunicorn.out.log
tail -f /var/log/jpmorgan/production.log
```

### Application Management

```bash
# Docker Compose
docker-compose -f docker-compose.production.yml restart app
docker-compose -f docker-compose.production.yml stop app
docker-compose -f docker-compose.production.yml start app

# Native Installation
sudo supervisorctl restart jpmorgan-api
sudo supervisorctl stop jpmorgan-api
sudo supervisorctl start jpmorgan-api
```

---

## 🔒 Security Checklist

- [ ] Changed all default passwords
- [ ] Generated secure SECRET_KEY and JWT keys
- [ ] Configured SSL/TLS certificates
- [ ] Enabled firewall (UFW or iptables)
- [ ] Set up rate limiting
- [ ] Configured CORS properly
- [ ] Enabled audit logging
- [ ] Set up automated backups
- [ ] Configured monitoring and alerting
- [ ] Reviewed and updated .env.production
- [ ] Disabled DEBUG mode
- [ ] Set up log rotation
- [ ] Configured database backups
- [ ] Tested disaster recovery procedures

---

## 🚨 Troubleshooting

### Application Won't Start

```bash
# Check logs
docker-compose logs app

# Check environment variables
docker-compose exec app env | grep DATABASE_URL

# Verify database connection
docker-compose exec app python -c "import psycopg2; psycopg2.connect('postgresql://...')"
```

### Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose ps postgresql

# Check PostgreSQL logs
docker-compose logs postgresql

# Test connection
docker-compose exec postgresql psql -U jpmorgan_prod -d jpmorgan_financial_apis_prod
```

### SSL Certificate Issues

```bash
# Verify certificate
openssl x509 -in nginx/ssl/server.crt -text -noout

# Check NGINX configuration
docker-compose exec nginx nginx -t

# Reload NGINX
docker-compose restart nginx
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Check application metrics
curl https://your-domain.com/metrics

# View Grafana dashboards
# Navigate to https://your-domain.com:3000
```

---

## 📈 Scaling

### Horizontal Scaling

```yaml
# In docker-compose.production.yml
services:
  app:
    deploy:
      replicas: 3
```

### Load Balancing

Update NGINX configuration to include multiple upstream servers:

```nginx
upstream jpmorgan_api {
    least_conn;
    server app1:8000;
    server app2:8000;
    server app3:8000;
}
```

---

## 🔄 Updates & Rollbacks

### Update Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.production.yml build app
docker-compose -f docker-compose.production.yml up -d app
```

### Rollback

```bash
# Stop current version
docker-compose -f docker-compose.production.yml down

# Restore from backup
tar -xzf /backups/backup_TIMESTAMP.tar.gz -C /opt/jpmorgan-financial-apis

# Restart
docker-compose -f docker-compose.production.yml up -d
```

---

## 📞 Support

### Logs Location
- Application: `/var/log/jpmorgan/`
- NGINX: `/var/log/nginx/`
- Docker: `docker-compose logs`

### Useful Commands

```bash
# View all running containers
docker-compose ps

# Execute command in container
docker-compose exec app bash

# View resource usage
docker stats

# Clean up old images
docker system prune -a
```

---

## ✅ Production Checklist

Before going live, ensure:

- [ ] All services are running and healthy
- [ ] SSL certificates are valid and auto-renewing
- [ ] Monitoring dashboards are accessible
- [ ] Backups are configured and tested
- [ ] Firewall rules are in place
- [ ] DNS records are configured
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation is up to date
- [ ] Team is trained on operations
- [ ] Incident response plan is ready
- [ ] Rollback procedure is tested

---

## 🎉 Success!

Your JPMorgan Financial APIs are now running in production!

**Access Points:**
- API: `https://your-domain.com`
- Health: `https://your-domain.com/health`
- Docs: `https://your-domain.com/docs`
- Grafana: `https://your-domain.com:3000`
- Prometheus: `https://your-domain.com:9090`

**Next Steps:**
1. Monitor application metrics
2. Set up alerting rules
3. Configure automated backups
4. Plan for scaling
5. Regular security audits

---

**Need Help?** Check the troubleshooting section or review the logs.
