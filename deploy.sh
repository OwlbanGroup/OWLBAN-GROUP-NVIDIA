#!/bin/bash

# JPMorgan Financial APIs - Automated Deployment Script
# This script automates the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOMAIN="api.equityshieldadvocates.com"
MAIN_DOMAIN="equityshieldadvocates.com"
EMAIL="admin@equityshieldadvocates.com"
DEPLOY_DIR="/opt/jpmorgan-api"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    error "This script should not be run as root"
    exit 1
fi

log "Starting JPMorgan Financial APIs deployment..."

# Step 1: Update system
log "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
success "System updated"

# Step 2: Install dependencies
log "Step 2: Installing required packages..."
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx git curl wget
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl start nginx
sudo systemctl enable nginx
sudo usermod -aG docker $USER
success "Dependencies installed"

# Step 3: Create deployment directory
log "Step 3: Setting up deployment directory..."
sudo mkdir -p $DEPLOY_DIR
sudo chown $USER:$USER $DEPLOY_DIR
cd $DEPLOY_DIR
success "Deployment directory created"

# Step 4: Setup project files
log "Step 4: Setting up project files..."
# Copy project files (assuming they're in the current directory)
# In production, this would be a git clone
mkdir -p logs data ssl backups

# Create environment file
cat > .env << EOF
FLASK_ENV=production
TESTING=0
DATABASE_URL=sqlite:///data/jpmorgan_api.db
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET_KEY=$(openssl rand -hex 32)
DOMAIN=$DOMAIN
MAIN_DOMAIN=$MAIN_DOMAIN
SSL_EMAIL=$EMAIL
EOF

chmod 600 .env
success "Environment configured"

# Step 5: SSL Certificate
log "Step 5: Setting up SSL certificates..."
sudo systemctl stop nginx

sudo certbot certonly --standalone \
  --email $EMAIL \
  --agree-tos \
  --no-eff-email \
  -d $DOMAIN

sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem ssl/
sudo chown $USER:$USER ssl/*.pem
chmod 600 ssl/*.pem

sudo systemctl start nginx
success "SSL certificates configured"

# Step 6: Deploy application
log "Step 6: Deploying application..."
docker-compose up -d --build
sleep 30

if docker-compose ps | grep -q "Up"; then
    success "Application deployed successfully"
else
    error "Application deployment failed"
    docker-compose logs
    exit 1
fi

# Step 7: Configure nginx
log "Step 7: Configuring nginx..."
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo nginx -t

if [ $? -eq 0 ]; then
    sudo systemctl reload nginx
    success "Nginx configured"
else
    error "Nginx configuration failed"
    exit 1
fi

# Step 8: Setup monitoring
log "Step 8: Setting up monitoring..."
chmod +x monitor.sh

sudo tee /etc/systemd/system/jpmorgan-monitor.service > /dev/null <<EOF
[Unit]
Description=JPMorgan API Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$DEPLOY_DIR
ExecStart=$DEPLOY_DIR/monitor.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jpmorgan-monitor
sudo systemctl start jpmorgan-monitor
success "Monitoring configured"

# Step 9: Setup SSL renewal
log "Step 9: Setting up SSL renewal..."
sudo crontab -l | { cat; echo "0 12 * * * /usr/bin/certbot renew --quiet && docker-compose restart nginx"; } | sudo crontab -
success "SSL renewal configured"

# Step 10: Final verification
log "Step 10: Running final verification..."

# Test health endpoint
if curl -f -s --max-time 10 https://$DOMAIN/health > /dev/null 2>&1; then
    success "Health check passed"
else
    warning "Health check failed - this may be normal if DNS hasn't propagated yet"
fi

# Test HTTPS
if curl -I -s https://$DOMAIN/ | head -1 | grep -q "200"; then
    success "HTTPS connection working"
else
    warning "HTTPS connection failed - check DNS propagation"
fi

# Display service status
echo
echo "=========================================="
echo "DEPLOYMENT SUMMARY"
echo "=========================================="
echo "Domain: https://$DOMAIN"
echo "Health Check: https://$DOMAIN/health"
echo "API Docs: https://$DOMAIN/api/docs"
echo "Dashboard: https://$DOMAIN/dashboard"
echo
echo "Service Status:"
docker-compose ps
echo
echo "Next Steps:"
echo "1. Configure DNS A record: api -> YOUR_SERVER_IP"
echo "2. Wait for DNS propagation (24-48 hours)"
echo "3. Test all endpoints with the demo script"
echo "4. Integrate with your main website"
echo "=========================================="

success "Deployment completed successfully!"
warning "Remember to configure DNS records as described in DNS_SETUP.md"
