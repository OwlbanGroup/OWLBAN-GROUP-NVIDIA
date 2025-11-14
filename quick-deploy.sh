#!/bin/bash

################################################################################
# JPMorgan Financial APIs - Quick Production Deployment
################################################################################
# One-command deployment script for production environment
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        JPMorgan Financial APIs - Production Deployment       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose first: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${BLUE}[1/6]${NC} Checking prerequisites..."
sleep 1

# Create necessary directories
echo -e "${BLUE}[2/6]${NC} Creating directories..."
mkdir -p logs backups nginx/ssl models
echo -e "${GREEN}✓${NC} Directories created"

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    echo -e "${YELLOW}[3/6]${NC} Creating production environment file..."
    if [ -f ".env.production.example" ]; then
        cp .env.production.example .env.production
        echo -e "${YELLOW}⚠${NC}  Please edit .env.production with your actual values"
        echo -e "${YELLOW}⚠${NC}  Press Enter when ready to continue..."
        read
    else
        echo -e "${RED}Error: .env.production.example not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}[3/6]${NC} Environment file exists"
fi

# Generate SSL certificates if they don't exist
if [ ! -f "nginx/ssl/server.crt" ] || [ ! -f "nginx/ssl/server.key" ]; then
    echo -e "${BLUE}[4/6]${NC} Generating SSL certificates..."
    python3 scripts/setup_https.py \
        --action generate-self-signed \
        --domain localhost \
        --cert-dir nginx/ssl \
        --days 365
    echo -e "${GREEN}✓${NC} SSL certificates generated"
    echo -e "${YELLOW}⚠${NC}  For production, replace with CA-signed certificates!"
else
    echo -e "${GREEN}[4/6]${NC} SSL certificates exist"
fi

# Build and start services
echo -e "${BLUE}[5/6]${NC} Building and starting services..."
docker-compose -f docker-compose.production.yml up -d --build

# Wait for services to be ready
echo -e "${BLUE}[6/6]${NC} Waiting for services to be ready..."
sleep 10

# Health check
echo ""
echo -e "${BLUE}Running health checks...${NC}"
if curl -f -k https://localhost/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Application is healthy"
else
    echo -e "${RED}✗${NC} Health check failed"
    echo "Check logs with: docker-compose -f docker-compose.production.yml logs app"
    exit 1
fi

# Display summary
echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║                  🎉 DEPLOYMENT SUCCESSFUL! 🎉                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${BLUE}Access Points:${NC}"
echo "  • API:        https://localhost"
echo "  • Health:     https://localhost/health"
echo "  • Docs:       https://localhost/docs"
echo "  • Grafana:    http://localhost:3000 (admin/SecureGrafanaP@ss2024)"
echo "  • Prometheus: http://localhost:9090"
echo ""
echo -e "${BLUE}Management Commands:${NC}"
echo "  • View logs:    docker-compose -f docker-compose.production.yml logs -f"
echo "  • Stop:         docker-compose -f docker-compose.production.yml stop"
echo "  • Restart:      docker-compose -f docker-compose.production.yml restart"
echo "  • Status:       docker-compose -f docker-compose.production.yml ps"
echo ""
echo -e "${YELLOW}Important Next Steps:${NC}"
echo "  1. Update SECRET_KEY in .env.production"
echo "  2. Replace SSL certificates with CA-signed certificates"
echo "  3. Configure your domain name"
echo "  4. Set up automated backups"
echo "  5. Configure monitoring alerts"
echo ""
echo -e "${GREEN}For detailed instructions, see: PRODUCTION_DEPLOYMENT_GUIDE.md${NC}"
echo ""
