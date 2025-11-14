#!/bin/bash

# JPMorgan Financial APIs - Health Check Script
# This script performs comprehensive health checks on the deployed application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:5000}"
TIMEOUT=30

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_http_endpoint() {
    local url=$1
    local expected_code=${2:-200}
    local description=$3

    log_info "Checking ${description}: ${url}"

    local response
    response=$(curl -s -w "HTTPSTATUS:%{http_code}" --max-time ${TIMEOUT} "${url}" 2>/dev/null)

    local body=$(echo "${response}" | sed 's/HTTPSTATUS.*//')
    local status_code=$(echo "${response}" | grep "HTTPSTATUS:" | sed 's/.*HTTPSTATUS://')

    if [ "${status_code}" = "${expected_code}" ]; then
        log_success "${description} is healthy (HTTP ${status_code})"
        return 0
    else
        log_error "${description} failed (HTTP ${status_code})"
        echo "Response: ${body}"
        return 1
    fi
}

check_database_health() {
    log_info "Checking database connectivity..."

    # Try to get database health from API
    if check_http_endpoint "${API_URL}/health/db" 200 "Database Health"; then
        log_success "Database is healthy"
        return 0
    else
        log_warning "Database health endpoint not available, checking via direct connection..."

        # If running in Docker Compose, check database directly
        if command -v docker-compose &> /dev/null && docker-compose ps | grep -q postgres; then
            if docker-compose exec -T postgres pg_isready -U telemetry_user -d telemetry_db > /dev/null 2>&1; then
                log_success "Database is healthy (direct connection)"
                return 0
            fi
        fi

        log_error "Database health check failed"
        return 1
    fi
}

check_redis_health() {
    log_info "Checking Redis connectivity..."

    # If running in Docker Compose, check Redis directly
    if command -v docker-compose &> /dev/null && docker-compose ps | grep -q redis; then
        if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
            log_success "Redis is healthy"
            return 0
        fi
    fi

    log_warning "Redis direct check not available, checking via API..."
    # Could add API endpoint for Redis health if implemented
    return 0
}

check_websocket_health() {
    log_info "Checking WebSocket connectivity..."

    # Use websocat or similar tool if available
    if command -v websocat &> /dev/null; then
        if timeout 10 websocat -E "ws://localhost:5000/ws" < /dev/null > /dev/null 2>&1; then
            log_success "WebSocket is healthy"
            return 0
        fi
    else
        log_warning "websocat not available, skipping WebSocket check"
        return 0
    fi

    log_error "WebSocket health check failed"
    return 1
}

check_api_endpoints() {
    log_info "Checking API endpoints..."

    local failed=0

    # Core endpoints
    check_http_endpoint "${API_URL}/health" 200 "API Health" || ((failed++))
    check_http_endpoint "${API_URL}/" 200 "Root Endpoint" || ((failed++))
    check_http_endpoint "${API_URL}/api/v1/telemetry/status" 200 "Telemetry Status" || ((failed++))
    check_http_endpoint "${API_URL}/data/formats" 200 "Data Formats" || ((failed++))

    # Authentication endpoints (may require tokens)
    check_http_endpoint "${API_URL}/auth/status" 401 "Auth Status (expected 401)" || ((failed++))

    # WebSocket status
    check_http_endpoint "${API_URL}/websocket/status" 200 "WebSocket Status" || ((failed++))

    if [ $failed -eq 0 ]; then
        log_success "All API endpoints are healthy"
        return 0
    else
        log_error "${failed} API endpoint(s) failed"
        return 1
    fi
}

check_performance() {
    log_info "Checking API performance..."

    # Measure response time for health endpoint
    local start_time=$(date +%s%N)
    if curl -s --max-time 10 "${API_URL}/health" > /dev/null; then
        local end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds

        if [ $duration -lt 1000 ]; then
            log_success "API response time: ${duration}ms (good)"
        elif [ $duration -lt 5000 ]; then
            log_warning "API response time: ${duration}ms (slow)"
        else
            log_error "API response time: ${duration}ms (very slow)"
            return 1
        fi
    else
        log_error "Performance check failed"
        return 1
    fi

    return 0
}

check_resource_usage() {
    log_info "Checking resource usage..."

    if command -v docker &> /dev/null && docker ps | grep -q jpmorgan; then
        log_info "Docker container resource usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    fi

    # Check disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ $disk_usage -gt 90 ]; then
        log_error "Disk usage is high: ${disk_usage}%"
        return 1
    elif [ $disk_usage -gt 80 ]; then
        log_warning "Disk usage is elevated: ${disk_usage}%"
    else
        log_success "Disk usage is normal: ${disk_usage}%"
    fi

    return 0
}

generate_report() {
    local report_file="health_report_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "JPMorgan Financial APIs - Health Check Report"
        echo "=========================================="
        echo "Timestamp: $(date)"
        echo "API URL: ${API_URL}"
        echo ""
        echo "Health Check Results:"
        echo "===================="
    } > "${report_file}"

    log_info "Health check report saved to ${report_file}"
}

# Main script
main() {
    echo "JPMorgan Financial APIs - Health Check"
    echo "====================================="

    local overall_status=0

    # Run all checks
    check_api_endpoints || overall_status=1
    check_database_health || overall_status=1
    check_redis_health || overall_status=1
    check_websocket_health || overall_status=1
    check_performance || overall_status=1
    check_resource_usage || overall_status=1

    echo ""
    if [ $overall_status -eq 0 ]; then
        log_success "All health checks passed!"
        generate_report
        exit 0
    else
        log_error "Some health checks failed. Please check the logs above."
        generate_report
        exit 1
    fi
}

# Parse command line arguments
case "${1:-all}" in
    "api")
        check_api_endpoints
        ;;
    "db")
        check_database_health
        ;;
    "redis")
        check_redis_health
        ;;
    "ws")
        check_websocket_health
        ;;
    "perf")
        check_performance
        ;;
    "resources")
        check_resource_usage
        ;;
    "all")
        main
        ;;
    *)
        echo "Usage: $0 {all|api|db|redis|ws|perf|resources}"
        echo ""
        echo "Commands:"
        echo "  all       - Run all health checks"
        echo "  api       - Check API endpoints"
        echo "  db        - Check database connectivity"
        echo "  redis     - Check Redis connectivity"
        echo "  ws        - Check WebSocket connectivity"
        echo "  perf      - Check API performance"
        echo "  resources - Check resource usage"
        exit 1
        ;;
esac
