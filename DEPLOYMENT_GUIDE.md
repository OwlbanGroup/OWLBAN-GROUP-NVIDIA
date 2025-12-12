# 🚀 JPMorgan Financial APIs - Complete Deployment Guide

## Integration with equityshieldadvocates.com

This guide provides complete instructions for deploying the JPMorgan Financial APIs as a subdomain (api.equityshieldadvocates.com) integrated with your existing website.

---

## 📋 Prerequisites

### System Requirements
- **Ubuntu 20.04+ or CentOS 7+** server
- **Root or sudo access** to the server
- **Domain name:** equityshieldadvocates.com configured
- **Static IP address** for the server
- **At least 2GB RAM, 20GB storage**

### Required Software
- Docker & Docker Compose
- Nginx
- Certbot (for SSL)
- Git
- Python 3.9+

### Network Requirements
- **Ports 80 and 443** open for HTTP/HTTPS
- **Port 22** open for SSH
- **Firewall configured** (ufw or firewalld)

---

## 🎯 Deployment Overview

### Architecture
```
Internet → Cloudflare/Namecheap DNS → Nginx (SSL Termination) → Docker Containers → Flask API
```

### Components Deployed
1. **Docker Container:** JPMorgan Financial APIs Flask application
2. **Nginx Reverse Proxy:** SSL termination, load balancing, security headers
3. **SSL Certificates:** Let's Encrypt automated certificates
4. **Monitoring:** Health checks and automated alerts
5. **DNS Configuration:** api.equityshieldadvocates.com subdomain

---

## 📝 Step-by-Step Deployment

### Step 1: Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx git curl wget

# Start and enable services
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl start nginx
sudo systemctl enable nginx

# Add user to docker group
sudo usermod -aG docker $USER

# Create deployment directory
sudo mkdir -p /opt/jpmorgan-api
sudo chown $USER:$USER /opt/jpmorgan-api
```

### Step 2: Clone and Configure Project

```bash
# Navigate to deployment directory
cd /opt/jpmorgan-api

# Clone project (replace with your repository)
git clone https://github.com/your-repo/jpmorgan-financial-apis.git .
# OR copy files from your local development environment

# Create necessary directories
mkdir -p logs data ssl

# Set proper permissions
chmod +x deploy.sh monitor.sh
chmod 755 logs data ssl
```

### Step 3: Environment Configuration

```bash
# Create environment file
cat > .env << EOF
FLASK_ENV=production
TESTING=0
DATABASE_URL=sqlite:///data/jpmorgan_api.db
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
DOMAIN=api.equityshieldadvocates.com
MAIN_DOMAIN=equityshieldadvocates.com
SSL_EMAIL=admin@equityshieldadvocates.com
EOF

# Secure the environment file
chmod 600 .env
```

### Step 4: DNS Configuration

Configure DNS records as described in `DNS_SETUP.md`:

1. **A Record:**
   ```
   Type: A
   Name: api
   Value: YOUR_SERVER_IP
   TTL: 300
   ```

2. **Wait for DNS propagation** (can take 24-48 hours)

3. **Verify DNS:**
   ```bash
   nslookup api.equityshieldadvocates.com
   ```

### Step 5: SSL Certificate Setup

```bash
# Stop nginx temporarily for certificate issuance
sudo systemctl stop nginx

# Get SSL certificate
sudo certbot certonly --standalone \
  --email admin@equityshieldadvocates.com \
  --agree-tos \
  --no-eff-email \
  -d api.equityshieldadvocates.com

# Copy certificates to project directory
sudo cp /etc/letsencrypt/live/api.equityshieldadvocates.com/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/api.equityshieldadvocates.com/privkey.pem ssl/

# Set proper permissions
sudo chown $USER:$USER ssl/*.pem
chmod 600 ssl/*.pem

# Start nginx
sudo systemctl start nginx
```

### Step 6: Deploy Application

```bash
# Build and start services
docker-compose up -d --build

# Wait for services to start
sleep 30

# Check service status
docker-compose ps

# Check application logs
docker-compose logs jpmorgan-api
```

### Step 7: Nginx Configuration

```bash
# Backup existing nginx configuration
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Copy our nginx configuration
sudo cp nginx.conf /etc/nginx/nginx.conf

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### Step 8: SSL Certificate Automation

```bash
# Create cron job for certificate renewal
sudo crontab -e

# Add this line:
0 12 * * * /usr/bin/certbot renew --quiet && docker-compose restart nginx
```

### Step 9: Monitoring Setup

```bash
# Make monitor script executable
chmod +x monitor.sh

# Create systemd service for monitoring
sudo tee /etc/systemd/system/jpmorgan-monitor.service > /dev/null <<EOF
[Unit]
Description=JPMorgan API Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/jpmorgan-api
ExecStart=/opt/jpmorgan-api/monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start monitoring
sudo systemctl daemon-reload
sudo systemctl enable jpmorgan-monitor
sudo systemctl start jpmorgan-monitor
```

### Step 10: Testing and Verification

```bash
# Test health endpoint
curl -k https://api.equityshieldadvocates.com/health

# Test API documentation
curl -I https://api.equityshieldadvocates.com/api/docs

# Test dashboard
curl -I https://api.equityshieldadvocates.com/dashboard

# Test SSL certificate
openssl s_client -connect api.equityshieldadvocates.com:443 -servername api.equityshieldadvocates.com < /dev/null
```

---

## 🔧 Integration with Main Website

### Option 1: Direct API Calls from Frontend

Add to your main website's JavaScript:

```javascript
// Example API integration
const API_BASE = 'https://api.equityshieldadvocates.com';

async function fetchFinancialData() {
    try {
        const response = await fetch(`${API_BASE}/api/financial-data`, {
            method: 'GET',
            headers: {
                'Authorization': 'Bearer YOUR_TOKEN',
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API call failed:', error);
    }
}
```

### Option 2: Server-Side Integration

If your main site uses PHP, Node.js, or Python:

```php
// PHP example
$api_url = 'https://api.equityshieldadvocates.com';
$token = 'YOUR_JWT_TOKEN';

$context = stream_context_create([
    'http' => [
        'method' => 'GET',
        'header' => "Authorization: Bearer $token\r\n" .
                   "Content-Type: application/json\r\n"
    ]
]);

$result = file_get_contents($api_url . '/api/businesses', false, $context);
$data = json_decode($result, true);
```

### Option 3: CORS Configuration

The nginx configuration already includes CORS headers for `equityshieldadvocates.com`. For additional domains, update the nginx.conf:

```nginx
add_header 'Access-Control-Allow-Origin' 'https://www.equityshieldadvocates.com, https://additional-domain.com' always;
```

---

## 📊 Monitoring and Maintenance

### Health Checks

```bash
# Manual health check
curl https://api.equityshieldadvocates.com/health

# Check service status
docker-compose ps

# View logs
docker-compose logs -f jpmorgan-api

# Monitor resource usage
docker stats
```

### Backup Strategy

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/jpmorgan-api/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker exec jpmorgan-api sqlite3 /app/data/jpmorgan_api.db .dump > $BACKUP_DIR/db_$DATE.sql

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz .env nginx.conf docker-compose.yml

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x backup.sh

# Add to cron for daily backups
echo "0 2 * * * /opt/jpmorgan-api/backup.sh" | crontab -
```

### Updates and Maintenance

```bash
# Update application
cd /opt/jpmorgan-api
git pull origin main
docker-compose build --no-cache
docker-compose up -d

# Update SSL certificates
sudo certbot renew

# Rotate logs
docker-compose logs --no-color > logs/app_$(date +%Y%m-%d).log
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. API Not Accessible
```bash
# Check if containers are running
docker-compose ps

# Check nginx configuration
sudo nginx -t

# Check firewall
sudo ufw status
```

#### 2. SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in ssl/fullchain.pem -text -noout

# Renew certificate manually
sudo certbot renew
```

#### 3. Database Issues
```bash
# Check database file
ls -la data/jpmorgan_api.db

# Check database integrity
docker exec jpmorgan-api sqlite3 /app/data/jpmorgan_api.db "PRAGMA integrity_check;"
```

#### 4. High Resource Usage
```bash
# Check container resources
docker stats

# Check system resources
htop

# Restart services
docker-compose restart
```

### Logs and Debugging

```bash
# Application logs
docker-compose logs jpmorgan-api

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log

# System logs
sudo journalctl -u jpmorgan-monitor -f
```

---

## 🔒 Security Checklist

- [ ] SSH key authentication enabled (no password login)
- [ ] Firewall configured (only ports 22, 80, 443 open)
- [ ] SSL certificates properly installed and auto-renewing
- [ ] Database file permissions secure (600)
- [ ] Environment variables not logged
- [ ] Rate limiting active
- [ ] Security headers configured
- [ ] Regular security updates scheduled
- [ ] Backup system operational

---

## 📞 Support and Resources

### Emergency Contacts
- **Technical Support:** admin@equityshieldadvocates.com
- **SSL Certificate Issues:** Let's Encrypt community forums
- **Docker Issues:** Docker documentation

### Useful Commands

```bash
# Quick status check
curl https://api.equityshieldadvocates.com/health && echo " - API OK"

# Full system status
docker-compose ps && sudo systemctl status nginx && sudo systemctl status jpmorgan-monitor

# Emergency restart
docker-compose down && docker-compose up -d && sudo systemctl restart nginx
```

### Documentation Links
- [Docker Documentation](https://docs.docker.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Let's Encrypt](https://letsencrypt.org/docs/)
- [Certbot](https://certbot.eff.org/docs/)

---

## ✅ Post-Deployment Checklist

- [ ] DNS records configured and propagated
- [ ] SSL certificates installed and working
- [ ] Docker containers running successfully
- [ ] Nginx reverse proxy configured
- [ ] API endpoints accessible via HTTPS
- [ ] Monitoring service active
- [ ] Backup system configured
- [ ] Main website integration tested
- [ ] Security hardening completed
- [ ] Documentation updated for users

---

**🎉 Deployment Complete!**

Your JPMorgan Financial APIs are now live at `https://api.equityshieldadvocates.com`

**Next Steps:**
1. ✅ Wait for DNS propagation (24-48 hours)
2. ✅ Test all endpoints using the demo script
3. ✅ Integrate API calls into your main website
4. ✅ Set up user accounts and authentication
5. ✅ Monitor system performance and logs

For any issues, refer to the troubleshooting section or contact technical support.
