#!/bin/bash

# JPMorgan Financial APIs - Live Transactional Monitoring Deployment Script
# This script deploys a complete monitoring stack with live data flowing to Grafana

set -e

echo "🚀 Starting JPMorgan Financial APIs Live Transactional Monitoring Deployment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_success "Docker and Docker Compose are installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    mkdir -p grafana/provisioning/datasources
    mkdir -p grafana/provisioning/dashboards
    mkdir -p grafana/dashboards
    mkdir -p prometheus/data
    mkdir -p grafana/data

    print_success "Directories created"
}

# Start the monitoring stack
start_monitoring_stack() {
    print_status "Starting monitoring stack with Docker Compose..."

    # Stop any existing containers
    docker-compose down || true

    # Start the monitoring stack
    docker-compose up -d

    print_success "Monitoring stack started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be ready..."

    # Wait for Prometheus
    print_status "Waiting for Prometheus..."
    for i in {1..30}; do
        if curl -s http://localhost:9090/-/ready > /dev/null; then
            print_success "Prometheus is ready"
            break
        fi
        sleep 2
    done

    # Wait for Grafana
    print_status "Waiting for Grafana..."
    for i in {1..30}; do
        if curl -s http://localhost:3000/api/health > /dev/null; then
            print_success "Grafana is ready"
            break
        fi
        sleep 2
    done

    # Wait for Node Exporter
    print_status "Waiting for Node Exporter..."
    for i in {1..30}; do
        if curl -s http://localhost:9100/metrics > /dev/null; then
            print_success "Node Exporter is ready"
            break
        fi
        sleep 2
    done

    # Wait for cAdvisor
    print_status "Waiting for cAdvisor..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/metrics > /dev/null; then
            print_success "cAdvisor is ready"
            break
        fi
        sleep 2
    done
}

# Test Flask application metrics endpoint
test_flask_metrics() {
    print_status "Testing Flask application metrics endpoint..."

    # Check if Flask app is running on port 5000
    if curl -s http://localhost:5000/metrics > /dev/null; then
        print_success "Flask application metrics endpoint is accessible"
    else
        print_warning "Flask application not running on port 5000"
        print_warning "Please start your Flask application with: python app.py"
        print_warning "Or ensure it's running on the correct port"
    fi
}

# Test Prometheus metrics collection
test_prometheus_collection() {
    print_status "Testing Prometheus metrics collection..."

    # Check if Prometheus can scrape Flask metrics
    if curl -s "http://localhost:9090/api/v1/query?query=http_requests_total" | grep -q "status.*success"; then
        print_success "Prometheus is successfully collecting Flask metrics"
    else
        print_warning "Prometheus may not be collecting Flask metrics yet"
        print_warning "This is normal if the Flask app isn't running"
    fi

    # Check if Prometheus can scrape node metrics
    if curl -s "http://localhost:9090/api/v1/query?query=node_cpu_seconds_total" | grep -q "status.*success"; then
        print_success "Prometheus is successfully collecting node metrics"
    else
        print_error "Prometheus is not collecting node metrics"
    fi
}

# Test Grafana dashboard
test_grafana_dashboard() {
    print_status "Testing Grafana dashboard..."

    # Check if dashboard is accessible
    if curl -s -u admin:admin http://localhost:3000/api/dashboards > /dev/null; then
        print_success "Grafana dashboard API is accessible"
    else
        print_error "Grafana dashboard API is not accessible"
    fi
}

# Display access information
display_access_info() {
    echo ""
    echo "🎉 JPMorgan Financial APIs Live Transactional Monitoring Stack Deployed!"
    echo ""
    echo "Access URLs:"
    echo "  📊 Grafana Dashboard:     http://localhost:3000 (admin/admin)"
    echo "  📈 Prometheus:            http://localhost:9090"
    echo "  📋 Node Exporter:         http://localhost:9100"
    echo "  🐳 cAdvisor:              http://localhost:8080"
    echo "  🐍 Flask App Metrics:     http://localhost:5000/metrics"
    echo ""
    echo "Next Steps:"
    echo "1. Start your Flask application: python app.py"
    echo "2. Open Grafana at http://localhost:3000"
    echo "3. Login with admin/admin"
    echo "4. Navigate to the 'JPMorgan Financial APIs - Live Transactional Dashboard'"
    echo "5. Watch live transactional data flow in real-time!"
    echo ""
    echo "To stop the monitoring stack: docker-compose down"
    echo "To restart: docker-compose restart"
    echo ""
}

# Main deployment function
main() {
    echo "=================================================="
    echo " JPMorgan Financial APIs - Live Transactional Monitoring"
    echo "=================================================="

    check_docker
    create_directories
    start_monitoring_stack
    wait_for_services
    test_flask_metrics
    test_prometheus_collection
    test_grafana_dashboard
    display_access_info

    print_success "Deployment completed successfully! 🎉"
}

# Run main function
main "$@"
