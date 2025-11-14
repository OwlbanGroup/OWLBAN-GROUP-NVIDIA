#!/bin/bash

# JPMorgan Financial APIs - Kubernetes Production Deployment Script
# This script deploys the application to a Kubernetes cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="default"
APP_NAME="jpmorgan-api"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-jpmorgan}"
DOCKER_TAG="${DOCKER_TAG:-latest}"

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

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi

    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi

    # Check if docker is installed (for building images)
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."

    # Build the image
    docker build -t ${DOCKER_REGISTRY}/financial-apis:${DOCKER_TAG} .

    # Push the image
    docker push ${DOCKER_REGISTRY}/financial-apis:${DOCKER_TAG}

    log_success "Docker image built and pushed: ${DOCKER_REGISTRY}/financial-apis:${DOCKER_TAG}"
}

update_image_tag() {
    log_info "Updating image tag in deployment manifest..."

    # Update the deployment.yaml with the new image tag
    sed -i "s|image: jpmorgan/financial-apis:latest|image: ${DOCKER_REGISTRY}/financial-apis:${DOCKER_TAG}|g" k8s/deployment.yaml

    log_success "Image tag updated in deployment manifest"
}

deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes cluster..."

    # Apply ConfigMap
    kubectl apply -f k8s/configmap.yaml

    # Apply Secret
    kubectl apply -f k8s/secret.yaml

    # Apply PostgreSQL
    kubectl apply -f k8s/postgres-deployment.yaml

    # Apply Redis
    kubectl apply -f k8s/redis-deployment.yaml

    # Apply API Deployment
    kubectl apply -f k8s/deployment.yaml

    # Apply Services
    kubectl apply -f k8s/service.yaml

    # Apply Ingress
    kubectl apply -f k8s/ingress.yaml

    log_success "All Kubernetes manifests applied"
}

wait_for_deployment() {
    log_info "Waiting for deployments to be ready..."

    # Wait for PostgreSQL
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n ${NAMESPACE}

    # Wait for Redis
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}

    # Wait for API
    kubectl wait --for=condition=available --timeout=300s deployment/${APP_NAME} -n ${NAMESPACE}

    log_success "All deployments are ready"
}

run_health_checks() {
    log_info "Running health checks..."

    # Get service URL
    SERVICE_IP=$(kubectl get svc jpmorgan-api-service -o jsonpath='{.spec.clusterIP}')
    SERVICE_PORT=$(kubectl get svc jpmorgan-api-service -o jsonpath='{.spec.ports[0].port}')

    # Wait for service to be ready
    sleep 30

    # Test health endpoint
    if kubectl run test-health --image=curlimages/curl --rm -i --restart=Never -- curl -f http://${SERVICE_IP}:${SERVICE_PORT}/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo ""
    echo "API Service:"
    kubectl get svc jpmorgan-api-service -n ${NAMESPACE}
    echo ""
    echo "Pods:"
    kubectl get pods -l app=jpmorgan-api -n ${NAMESPACE}
    echo ""
    echo "Ingress:"
    kubectl get ingress jpmorgan-api-ingress -n ${NAMESPACE}
    echo ""
    log_success "Deployment completed successfully!"
}

rollback_deployment() {
    log_warning "Rolling back deployment..."

    # Rollback the deployment
    kubectl rollout undo deployment/${APP_NAME} -n ${NAMESPACE}

    # Wait for rollback to complete
    kubectl rollout status deployment/${APP_NAME} -n ${NAMESPACE}

    log_success "Rollback completed"
}

cleanup() {
    log_info "Cleaning up test resources..."
    kubectl delete pod test-health --ignore-not-found=true -n ${NAMESPACE}
}

# Main script
main() {
    echo "JPMorgan Financial APIs - Kubernetes Production Deployment"
    echo "======================================================"

    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_and_push_image
            update_image_tag
            deploy_to_kubernetes
            wait_for_deployment
            run_health_checks
            show_deployment_info
            cleanup
            ;;
        "rollback")
            rollback_deployment
            ;;
        "status")
            show_deployment_info
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "Usage: $0 {deploy|rollback|status|cleanup}"
            exit 1
            ;;
    esac
}

# Trap cleanup function
trap cleanup EXIT

# Run main function
main "$@"
