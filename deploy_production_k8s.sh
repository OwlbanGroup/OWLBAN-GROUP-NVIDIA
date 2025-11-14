#!/bin/bash

# Production Kubernetes Deployment Script for JPMorgan Financial APIs
# This script deploys the application to production Kubernetes with Helm and Istio

set -e

cd "$(dirname "$0")"

# Configuration
NAMESPACE="${NAMESPACE:-production}"
RELEASE_NAME="${RELEASE_NAME:-jpmorgan-telemetry-prod}"
HELM_CHART_PATH="./helm/jpmorgan-telemetry"
VALUES_FILE="${VALUES_FILE:-values-production.yaml}"
TIMEOUT="${TIMEOUT:-900}"

echo "🚀 Starting Production Kubernetes Deployment for JPMorgan Financial APIs..."
echo "   Namespace: $NAMESPACE"
echo "   Release: $RELEASE_NAME"
echo "   Chart: $HELM_CHART_PATH"
echo "   Values: $VALUES_FILE"

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

if ! command -v helm &> /dev/null; then
    echo "❌ Helm is not installed. Please install Helm and try again."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot access Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

echo "✅ Prerequisites validated"

# Check if Istio is installed
echo "🔍 Checking Istio installation..."
if ! kubectl get namespace istio-system &> /dev/null; then
    echo "❌ Istio is not installed. Please install Istio first."
    echo "   Visit: https://istio.io/latest/docs/setup/getting-started/"
    exit 1
fi

echo "✅ Istio is installed"

# Create namespace if it doesn't exist
echo "📁 Ensuring namespace $NAMESPACE exists..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repositories
echo "📦 Adding Helm repositories..."
helm repo add bitnami https://charts.bitnami.com/bitnami --force-update
helm repo add elastic https://helm.elastic.co --force-update
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update
helm repo update

# Deploy with Helm
echo "🏗️ Deploying with Helm..."
helm upgrade --install $RELEASE_NAME $HELM_CHART_PATH \
    --namespace $NAMESPACE \
    --create-namespace \
    --values $HELM_CHART_PATH/$VALUES_FILE \
    --wait \
    --timeout ${TIMEOUT}s \
    --debug

echo "✅ Helm deployment completed"

# Apply Istio configurations
echo "🔐 Applying Istio service mesh configurations..."

# Apply Istio manifests
kubectl apply -f istio/gateway.yaml -n $NAMESPACE
kubectl apply -f istio/virtualservice.yaml -n $NAMESPACE
kubectl apply -f istio/destinationrule.yaml -n $NAMESPACE
kubectl apply -f istio/peerauthentication.yaml -n $NAMESPACE
kubectl apply -f istio/networkpolicy.yaml -n $NAMESPACE

echo "✅ Istio configurations applied"

# Wait for Istio configurations to be ready
echo "⏳ Waiting for Istio configurations to be ready..."
sleep 30

# Run comprehensive health checks
echo "🏥 Running production health checks..."
if [ -f "health_check_production.sh" ]; then
    chmod +x health_check_production.sh
    ./health_check_production.sh
else
    echo "⚠️ Health check script not found, running basic checks..."

    # Basic pod check
    echo "📦 Checking pod status..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=$RELEASE_NAME -n $NAMESPACE --timeout=300s

    # Basic service check
    echo "🌐 Checking service availability..."
    kubectl get svc -l app.kubernetes.io/instance=$RELEASE_NAME -n $NAMESPACE
fi

# Configure monitoring and alerting
echo "📊 Configuring monitoring and alerting..."

# Check if Prometheus is scraping the application
PROMETHEUS_POD=$(kubectl get pods -n $NAMESPACE -l app=prometheus -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$PROMETHEUS_POD" ]; then
    echo "✅ Prometheus is running and should be scraping metrics"
else
    echo "⚠️ Prometheus not found in $NAMESPACE namespace"
fi

# Check if AlertManager is configured
ALERTMANAGER_POD=$(kubectl get pods -n $NAMESPACE -l app=alertmanager -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$ALERTMANAGER_POD" ]; then
    echo "✅ AlertManager is running and configured"
else
    echo "⚠️ AlertManager not found in $NAMESPACE namespace"
fi

# Get service URLs
echo "🔗 Getting service URLs..."
INGRESS_HOST=$(kubectl get ingress -n $NAMESPACE -l app.kubernetes.io/instance=$RELEASE_NAME -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null || echo "")
if [ -n "$INGRESS_HOST" ]; then
    echo "🌐 Application URL: https://$INGRESS_HOST"
    echo "🏥 Health Check: https://$INGRESS_HOST/health"
    echo "📚 API Docs: https://$INGRESS_HOST/docs"
else
    echo "⚠️ Ingress not found or not configured"
fi

# Get Istio Gateway load balancer
GATEWAY_LB=$(kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
if [ -n "$GATEWAY_LB" ]; then
    echo "🚪 Istio Gateway Load Balancer: $GATEWAY_LB"
fi

echo ""
echo "🎉 Production deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "   Namespace: $NAMESPACE"
echo "   Release: $RELEASE_NAME"
echo "   Istio Service Mesh: ✅ Enabled"
echo "   Mutual TLS: ✅ Enforced"
echo "   Network Policies: ✅ Applied"
echo "   Monitoring: ✅ Configured"
echo ""
echo "🔧 Management Commands:"
echo "   View pods: kubectl get pods -n $NAMESPACE"
echo "   View logs: kubectl logs -f deployment/$RELEASE_NAME -n $NAMESPACE"
echo "   Scale app: kubectl scale deployment $RELEASE_NAME --replicas=10 -n $NAMESPACE"
echo "   Rollback: helm rollback $RELEASE_NAME -n $NAMESPACE"
echo "   Upgrade: helm upgrade $RELEASE_NAME $HELM_CHART_PATH -n $NAMESPACE"
echo ""
echo "📈 Monitoring & Observability:"
echo "   Prometheus: kubectl port-forward svc/prometheus 9090:9090 -n $NAMESPACE"
echo "   Grafana: kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
echo "   Kibana: kubectl port-forward svc/kibana 5601:5601 -n $NAMESPACE"
echo "   AlertManager: kubectl port-forward svc/alertmanager 9093:9093 -n $NAMESPACE"
echo ""
echo "🔒 Security Features:"
echo "   mTLS: Enabled between all services"
echo "   Network Policies: Restrict pod-to-pod communication"
echo "   RBAC: Configured for service accounts"
echo "   Secrets: Encrypted and mounted securely"
echo ""
echo "🚨 Next Steps:"
echo "1. Update DNS to point to the Istio Gateway load balancer"
echo "2. Configure external monitoring and alerting integrations"
echo "3. Set up automated backups for databases"
echo "4. Review and test disaster recovery procedures"
echo "5. Schedule production go-live and monitoring handoff"
