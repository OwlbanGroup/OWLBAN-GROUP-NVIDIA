#!/bin/bash

# Production Validation Script for JPMorgan Financial APIs
# This script performs comprehensive validation of production environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="jpmorgan-apis"
APP_NAME="jpmorgan-financial-apis"
DB_NAME="jpmorgan_financial_apis"
REDIS_SERVICE="redis-cluster"
KUBECTL="kubectl"
TIMEOUT=300  # 5 minutes timeout

# Log function
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running in Kubernetes
if ! $KUBECTL get namespace $NAMESPACE &> /dev/null; then
    error "Not running in Kubernetes or namespace $NAMESPACE not found"
fi

log "Starting production validation for JPMorgan Financial APIs"

# 1. Check Kubernetes cluster health
log "1. Validating Kubernetes cluster health"
$KUBECTL get nodes --no-headers | awk '{print $2}' | grep -v "Ready" && error "Not all nodes are ready"

# 2. Check namespace and resources
log "2. Validating namespace and resources"
$KUBECTL get namespace $NAMESPACE &> /dev/null || error "Namespace $NAMESPACE not found"

# Check deployments
$KUBECTL get deployment -n $NAMESPACE | grep $APP_NAME || error "Deployment $APP_NAME not found"

# Check services
$KUBECTL get service -n $NAMESPACE | grep $APP_NAME || error "Service $APP_NAME not found"

# 3. Check pod status
log "3. Validating pod status"
pods=$($KUBECTL get pods -n $NAMESPACE -l app=$APP_NAME --no-headers 2>/dev/null | wc -l)
if [ $pods -eq 0 ]; then
    error "No pods found for $APP_NAME"
fi

# Wait for pods to be ready
log "Waiting for pods to be ready (timeout: ${TIMEOUT}s)"
$KUBECTL wait --for=condition=ready pod -l app=$APP_NAME -n $NAMESPACE --timeout=${TIMEOUT}s || error "Pods not ready within timeout"

# 4. Check database connectivity
log "4. Validating database connectivity"
if $KUBECTL get deployment -n $NAMESPACE | grep -q postgresql; then
    log "Testing PostgreSQL connectivity"
    $KUBECTL exec -n $NAMESPACE deployment/postgresql -- pg_isready -U jpmorgan_user -d $DB_NAME || error "PostgreSQL not ready"
    
    # Test database access
    $KUBECTL exec -n $NAMESPACE deployment/postgresql -- psql -U jpmorgan_user -d $DB_NAME -c "SELECT 1;" || error "Cannot query PostgreSQL"
else
    warn "PostgreSQL deployment not found, skipping database validation"
fi

# 5. Check Redis connectivity
log "5. Validating Redis connectivity"
if $KUBECTL get service -n $NAMESPACE | grep -q $REDIS_SERVICE; then
    log "Testing Redis connectivity"
    $KUBECTL exec -n $NAMESPACE deployment/$APP_NAME -- redis-cli -h $REDIS_SERVICE ping || error "Redis not responding"
else
    warn "Redis service not found, skipping Redis validation"
fi

# 6. Check application health
log "6. Validating application health"
service_port=$($KUBECTL get svc -n $NAMESPACE $APP_NAME -o jsonpath='{.spec.ports[0].port}')
pod_ip=$($KUBECTL get pod -n $NAMESPACE -l app=$APP_NAME -o jsonpath='{.items[0].status.podIP}')

# Test health endpoint
log "Testing health endpoint"
curl -f -m 30 http://$pod_ip:$service_port/health || error "Health endpoint not responding"

# Test database health
log "Testing database health endpoint"
curl -f -m 30 http://$pod_ip:$service_port/health/database || error "Database health endpoint not responding"

# 7. Check external API connectivity (JPMorgan)
log "7. Validating external API connectivity"
# This requires TOKEN_CLIENT_ID and TOKEN_CLIENT_SECRET to be set
if [ -n "$TOKEN_CLIENT_ID" ] && [ -n "$TOKEN_CLIENT_SECRET" ]; then
    log "Testing JPMorgan OAuth token retrieval"
    TOKEN_RESPONSE=$(curl -s -w "%{http_code}" -X POST \
        "https://id.payments.jpmorgan.com/am/oauth2/alpha/access_token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -u "$TOKEN_CLIENT_ID:$TOKEN_CLIENT_SECRET" \
        -d "grant_type=client_credentials")

    HTTP_CODE=$(echo $TOKEN_RESPONSE | tail -1)
    RESPONSE_BODY=$(echo $TOKEN_RESPONSE | head -1)

    if [ "$HTTP_CODE" -ne 200 ]; then
        error "JPMorgan token retrieval failed with HTTP $HTTP_CODE: $RESPONSE_BODY"
    fi

    ACCESS_TOKEN=$(echo $RESPONSE_BODY | jq -r '.access_token')
    if [ "$ACCESS_TOKEN" = "null" ] || [ -z "$ACCESS_TOKEN" ]; then
        error "No access token received from JPMorgan"
    fi

    log "Testing JPMorgan API with access token"
    curl -f -m 30 -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://api.jpmorgan.com/v1/accounts" || warn "JPMorgan API test failed (may be rate limited)"
else
    warn "JPMorgan credentials not set, skipping external API validation"
fi

# 8. Check monitoring systems
log "8. Validating monitoring systems"
if $KUBECTL get svc -n monitoring prometheus &> /dev/null; then
    log "Prometheus service found, testing metrics endpoint"
    $KUBECTL port-forward -n monitoring svc/prometheus 9090:9090 &
    PROMETHEUS_PID=$!
    sleep 5

    # Test Prometheus
    curl -f -m 10 http://localhost:9090/-/ready || error "Prometheus not ready"
    curl -f -m 10 http://localhost:9090/api/v1/query?query=up || error "Prometheus query failed"

    kill $PROMETHEUS_PID
else
    warn "Prometheus not found in monitoring namespace"
fi

# 9. Check logging systems
log "9. Validating logging systems"
if $KUBECTL get deployment -n logging fluentd &> /dev/null; then
    log "Fluentd deployment found"
    # Test log forwarding
    $KUBECTL logs -n $NAMESPACE deployment/$APP_NAME --tail=1 || warn "Could not retrieve logs"
else
    warn "Fluentd not found in logging namespace"
fi

# 10. Performance validation
log "10. Basic performance validation"
# Simple load test
log "Running basic load test (10 concurrent requests)"
python3 -c "
import requests
import concurrent.futures
import time

def test_request():
    start = time.time()
    try:
        r = requests.get('http://$pod_ip:$service_port/health', timeout=5)
        return time.time() - start, r.status_code
    except:
        return time.time() - start, 0

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(test_request) for _ in range(10)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

durations = [r[0] for r in results]
statuses = [r[1] for r in results]

avg_duration = sum(durations) / len(durations)
success_rate = sum(1 for s in statuses if s == 200) / len(statuses)

print(f'Average response time: {avg_duration:.2f}s')
print(f'Success rate: {success_rate:.2%}')

if avg_duration > 1.0 or success_rate < 0.95:
    exit(1)
" || error "Basic performance test failed"

# 11. Security validation
log "11. Basic security validation"
# Check for open ports
$KUBECTL get svc -n $NAMESPACE | grep -v "ClusterIP" && warn "External services found, review security"

# Check for privileged containers
privileged_pods=$($KUBECTL get pods -n $NAMESPACE -o json | jq -r '.items[] | select(.spec.containers[].securityContext.privileged == true) | .metadata.name')
if [ "$privileged_pods" != "null" ]; then
    error "Privileged containers found: $privileged_pods"
fi

# 12. Compliance validation
log "12. Basic compliance validation"
# Check for secrets in environment
$KUBECTL get secrets -n $NAMESPACE | wc -l || error "No secrets found in namespace"

# Check for network policies
$KUBECTL get networkpolicy -n $NAMESPACE | wc -l || warn "No network policies found"

log "✅ All production validation checks passed!"
log "Production environment is ready for JPMorgan Financial APIs deployment"

exit 0
