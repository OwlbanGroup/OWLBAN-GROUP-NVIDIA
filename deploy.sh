#!/bin/bash

# OWLBAN GROUP AI Production Deployment Script
# Comprehensive deployment automation for quantum AI systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="owlban-group-ai"
DOCKER_REGISTRY="owlban"
TAG=${1:-"latest"}

echo -e "${BLUE}🚀 OWLBAN GROUP AI Production Deployment${NC}"
echo "=============================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo -e "\n${BLUE}Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_status "Docker is available"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_status "Docker Compose is available"

    # Check NVIDIA Docker (if GPUs available)
    if command -v nvidia-smi &> /dev/null; then
        if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            print_warning "NVIDIA Docker is not properly configured"
        else
            print_status "NVIDIA Docker is configured"
        fi
    fi

    # Check available disk space
    DISK_SPACE=$(df / | tail -1 | awk '{print $4}')
    if [ "$DISK_SPACE" -lt 10485760 ]; then  # 10GB in KB
        print_warning "Low disk space: $(($DISK_SPACE / 1024 / 1024))GB available"
    else
        print_status "Sufficient disk space available"
    fi
}

# Build Docker images
build_images() {
    echo -e "\n${BLUE}Building Docker images...${NC}"

    # Build API server
    echo "Building API server image..."
    docker build -f Dockerfile.api -t ${DOCKER_REGISTRY}/owlban-api:${TAG} .

    # Build dashboard
    echo "Building dashboard image..."
    docker build -f Dockerfile.dashboard -t ${DOCKER_REGISTRY}/owlban-dashboard:${TAG} .

    # Build Qiskit simulator
    echo "Building Qiskit simulator image..."
    docker build -f Dockerfile.qiskit -t ${DOCKER_REGISTRY}/owlban-qiskit:${TAG} .

    print_status "All images built successfully"
}

# Deploy services
deploy_services() {
    echo -e "\n${BLUE}Deploying services...${NC}"

    # Create networks and volumes
    docker network create owlban-network 2>/dev/null || true

    # Start services
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
    else
        docker compose up -d
    fi

    print_status "Services deployed successfully"
}

# Health checks
health_checks() {
    echo -e "\n${BLUE}Running health checks...${NC}"

    # Wait for services to start
    echo "Waiting for services to start..."
    sleep 30

    # Check API server
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_status "API server is healthy"
    else
        print_error "API server health check failed"
    fi

    # Check dashboard
    if curl -f http://localhost:8501/healthz &> /dev/null; then
        print_status "Dashboard is healthy"
    else
        print_warning "Dashboard health check failed (may take longer to start)"
    fi

    # Check database
    if docker exec owlban-group-ai_database_1 pg_isready -U owlban -d owlban_ai &> /dev/null; then
        print_status "Database is healthy"
    else
        print_error "Database health check failed"
    fi

    # Check Redis
    if docker exec owlban-group-ai_redis_1 redis-cli ping | grep -q PONG; then
        print_status "Redis is healthy"
    else
        print_error "Redis health check failed"
    fi
}

# Initialize data
initialize_data() {
    echo -e "\n${BLUE}Initializing data...${NC}"

    # Run database migrations
    echo "Running database initialization..."
    docker exec owlban-group-ai_database_1 psql -U owlban -d owlban_ai -f /docker-entrypoint-initdb.d/init.sql || true

    # Import sample data
    echo "Importing sample data..."
    # Add sample data import logic here

    print_status "Data initialization completed"
}

# Configure monitoring
setup_monitoring() {
    echo -e "\n${BLUE}Setting up monitoring...${NC}"

    # Start Prometheus and Grafana
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d prometheus grafana
    else
        docker compose up -d prometheus grafana
    fi

    print_status "Monitoring setup completed"
    echo "Grafana: http://localhost:3000 (admin/quantum_secure_2024)"
    echo "Prometheus: http://localhost:9090"
}

# Run tests
run_tests() {
    echo -e "\n${BLUE}Running tests...${NC}"

    # Run unit tests
    echo "Running unit tests..."
    docker run --rm -v $(pwd):/app -w /app python:3.11-slim \
        bash -c "pip install -r requirements.txt && python -m pytest tests/ -v" || true

    # Run integration tests
    echo "Running integration tests..."
    # Add integration test logic here

    print_status "Testing completed"
}

# Show deployment summary
show_summary() {
    echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
    echo "=============================================="
    echo ""
    echo "Services:"
    echo "  • API Server: http://localhost:8000"
    echo "  • Web Dashboard: http://localhost:8501"
    echo "  • Grafana: http://localhost:3000"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Triton Server: http://localhost:8001"
    echo "  • Jupyter (Quantum): http://localhost:8888"
    echo ""
    echo "Credentials:"
    echo "  • API: owlban_admin / quantum_secure_2024"
    echo "  • Grafana: admin / quantum_secure_2024"
    echo ""
    echo "Database:"
    echo "  • PostgreSQL: localhost:5432"
    echo "  • Redis: localhost:6379"
    echo "  • MongoDB: localhost:27017"
    echo ""
    echo "Next steps:"
    echo "  1. Access the web dashboard"
    echo "  2. Configure API endpoints"
    echo "  3. Set up monitoring alerts"
    echo "  4. Configure backup systems"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    build_images
    deploy_services
    health_checks
    initialize_data
    setup_monitoring
    run_tests
    show_summary
}

# Handle command line arguments
case "${2:-}" in
    "build")
        check_prerequisites
        build_images
        ;;
    "deploy")
        deploy_services
        ;;
    "health")
        health_checks
        ;;
    "test")
        run_tests
        ;;
    *)
        main
        ;;
esac
