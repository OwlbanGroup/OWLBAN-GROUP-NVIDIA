#!/bin/bash

# JPMorgan Financial APIs - Production Health Check Script
# This script performs comprehensive health checks for production deployment

set -e

NAMESPACE="${NAMESPACE:-production}"
RELEASE_NAME="${RELEASE_NAME:-jpmorgan-telemetry-prod}"
TIMEOUT="${TIMEOUT:-300}"

echo "🔍 Starting production health checks for $RELEASE_NAME in namespace $NAMESPACE"

# Function to check pod status
check_pods() {
    echo "📦 Checking pod status..."
    local pods_ready=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME --no-headers | grep -c "Running")
    local total_pods=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME --no-headers | wc -l)

    if [ "$pods_ready" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
        echo "✅ All $pods_ready pods are running"
        return 0
    else
        echo "❌ Pod status check failed: $pods_ready/$total_pods pods running"
        return 1
    fi
}
export -f check_pods

# Function to check service endpoints
check_services() {
    echo "🌐 Checking service endpoints..."
    local service_ip=$(kubectl get svc -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -o jsonpath='{.items[0].spec.clusterIP}')

    if [ -n "$service_ip" ]; then
        echo "✅ Service is available at $service_ip"
        return 0
    else
        echo "❌ Service check failed"
        return 1
    fi
}
export -f check_services

# Function to check application health endpoint
check_application_health() {
    echo "🏥 Checking application health endpoint..."
    local pod_name=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -o jsonpath='{.items[0].metadata.name}')

    if [ -n "$pod_name" ]; then
        # Wait for pod to be ready
        kubectl wait --for=condition=ready pod/$pod_name -n $NAMESPACE --timeout=60s

        # Check health endpoint
        local health_status=$(kubectl exec -n $NAMESPACE $pod_name -- curl -f http://localhost:8001/health 2>/dev/null || echo "failed")

        if [ "$health_status" != "failed" ]; then
            echo "✅ Application health check passed"
            return 0
        else
            echo "❌ Application health check failed"
            return 1
        fi
    else
        echo "❌ No application pods found"
        return 1
    fi
}
export -f check_application_health

# Function to check database connectivity
check_database() {
    echo "🗄️ Checking database connectivity..."
    local pod_name=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -o jsonpath='{.items[0].metadata.name}')

    if [ -n "$pod_name" ]; then
        # Check database connection via application
        local db_status=$(kubectl exec -n $NAMESPACE $pod_name -- python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgresql'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB', 'telemetry')
    )
    conn.close()
    print('connected')
except Exception as e:
    print('failed')
" 2>/dev/null)

        if [ "$db_status" = "connected" ]; then
            echo "✅ Database connectivity check passed"
            return 0
        else
            echo "❌ Database connectivity check failed"
            return 1
        fi
    else
        echo "❌ No application pods found for database check"
        return 1
    fi
}
export -f check_database

# Function to check Redis connectivity
check_redis() {
    echo "🔴 Checking Redis connectivity..."
    local pod_name=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -o jsonpath='{.items[0].metadata.name}')

    if [ -n "$pod_name" ]; then
        local redis_status=$(kubectl exec -n $NAMESPACE $pod_name -- python -c "
import redis
try:
    r = redis.Redis(host='redis', port=6379, password=os.getenv('REDIS_PASSWORD'))
    r.ping()
    print('connected')
except Exception as e:
    print('failed')
" 2>/dev/null)

        if [ "$redis_status" = "connected" ]; then
            echo "✅ Redis connectivity check passed"
            return 0
        else
            echo "❌ Redis connectivity check failed"
            return 1
        fi
    else
        echo "❌ No application pods found for Redis check"
        return 1
    fi
}
export -f check_redis

# Function to check monitoring stack
check_monitoring() {
    echo "📊 Checking monitoring stack..."

    # Check Prometheus
    local prom_pods=$(kubectl get pods -n $NAMESPACE -l app=prometheus --no-headers | grep -c "Running")
    if [ "$prom_pods" -gt 0 ]; then
        echo "✅ Prometheus is running ($prom_pods pods)"
    else
        echo "❌ Prometheus check failed"
        return 1
    fi

    # Check Elasticsearch
    local es_pods=$(kubectl get pods -n $NAMESPACE -l app=elasticsearch --no-headers | grep -c "Running")
    if [ "$es_pods" -gt 0 ]; then
        echo "✅ Elasticsearch is running ($es_pods pods)"
    else
        echo "❌ Elasticsearch check failed"
        return 1
    fi

    return 0
}
export -f check_monitoring

# Function to check Istio configuration
check_istio() {
    echo "🔐 Checking Istio configuration..."

    # Check if Istio is enabled and configured
    local istio_enabled=$(kubectl get peerauthentication -n $NAMESPACE -o name | wc -l)
    if [ "$istio_enabled" -gt 0 ]; then
        echo "✅ Istio PeerAuthentication is configured"
    else
        echo "⚠️ Istio PeerAuthentication not found"
    fi

    local gateway_count=$(kubectl get gateway -n $NAMESPACE -o name | wc -l)
    if [ "$gateway_count" -gt 0 ]; then
        echo "✅ Istio Gateway is configured"
    else
        echo "❌ Istio Gateway not found"
        return 1
    fi

    local vs_count=$(kubectl get virtualservice -n $NAMESPACE -o name | wc -l)
    if [ "$vs_count" -gt 0 ]; then
        echo "✅ Istio VirtualService is configured"
    else
        echo "❌ Istio VirtualService not found"
        return 1
    fi

    return 0
}
export -f check_istio

# Function to run load test validation
run_load_test_validation() {
    echo "🚀 Running load test validation..."

    if [ -d "load-testing" ]; then
        cd load-testing
        npm install > /dev/null 2>&1

        # Run smoke test
        if npm run test:smoke; then
            echo "✅ Load test smoke validation passed"
            return 0
        else
            echo "❌ Load test smoke validation failed"
            return 1
        fi
    else
        echo "⚠️ Load testing directory not found, skipping validation"
        return 0
    fi
}
export -f run_load_test_validation

# Main execution
echo "⏰ Starting health checks with $TIMEOUT second timeout..."

# Run all checks with timeout
timeout $TIMEOUT bash <<EOF
$(declare -f check_pods)
$(declare -f check_services)
$(declare -f check_application_health)
$(declare -f check_database)
$(declare -f check_redis)
$(declare -f check_monitoring)
$(declare -f check_istio)
$(declare -f run_load_test_validation)

check_pods
check_services
check_application_health
check_database
check_redis
check_monitoring
check_istio
run_load_test_validation
EOF

if [ $? -ne 0 ]; then
    echo "❌ Health checks timed out or failed"
    exit 1
fi

echo ""
echo "🎉 All production health checks passed!"
echo "✅ Deployment is ready for production traffic"
echo ""
echo "📋 Next steps:"
echo "1. Update DNS to point to the load balancer"
echo "2. Configure monitoring alerts"
echo "3. Set up backup schedules"
echo "4. Update incident response procedures"
echo "5. Schedule production go-live"
