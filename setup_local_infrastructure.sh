#!/bin/bash
# JPMorgan Financial APIs - Local Infrastructure Setup Script
# This script sets up local infrastructure for testing production deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if running on supported OS
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Detected Linux OS"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_info "Detected macOS"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        log_info "Detected Windows (using WSL or similar)"
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Install Docker
install_docker() {
    log_info "Installing Docker..."

    if command -v docker >/dev/null 2>&1; then
        log_info "Docker is already installed"
        return
    fi

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Install Docker on Linux
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
        log_success "Docker installed on Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Install Docker Desktop on macOS
        log_warning "Please install Docker Desktop for macOS manually from https://www.docker.com/products/docker-desktop"
        log_info "After installation, run: open /Applications/Docker.app"
    else
        log_warning "Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop"
    fi
}

# Start Docker service
start_docker() {
    log_info "Starting Docker service..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start docker
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open /Applications/Docker.app
        sleep 10  # Wait for Docker to start
    fi

    # Wait for Docker to be ready
    log_info "Waiting for Docker to be ready..."
    while ! docker info >/dev/null 2>&1; do
        sleep 2
    done

    log_success "Docker is ready"
}

# Install kubectl
install_kubectl() {
    log_info "Installing kubectl..."

    if command -v kubectl >/dev/null 2>&1; then
        log_info "kubectl is already installed"
        return
    fi

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
        chmod +x kubectl
        sudo mv kubectl /usr/local/bin/
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install kubectl
    else
        log_warning "Please install kubectl manually for Windows"
    fi

    log_success "kubectl installed"
}

# Install Minikube
install_minikube() {
    log_info "Installing Minikube..."

    if command -v minikube >/dev/null 2>&1; then
        log_info "Minikube is already installed"
        return
    fi

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
        sudo install minikube-linux-amd64 /usr/local/bin/minikube
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install minikube
    else
        log_warning "Please install Minikube manually for Windows"
    fi

    log_success "Minikube installed"
}

# Start Minikube cluster
start_minikube() {
    log_info "Starting Minikube cluster..."

    # Start Minikube with necessary configurations
    minikube start --driver=docker --cpus=2 --memory=4096 --kubernetes-version=v1.25.0

    # Enable ingress
    minikube addons enable ingress

    # Enable metrics-server for HPA
    minikube addons enable metrics-server

    # Wait for cluster to be ready
    kubectl wait --for=condition=ready node/minikube --timeout=300s

    log_success "Minikube cluster is ready"
}

# Install Helm
install_helm() {
    log_info "Installing Helm..."

    if command -v helm >/dev/null 2>&1; then
        log_info "Helm is already installed"
        return
    fi

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl https://get.helm.sh/helm-v3.10.0-linux-amd64.tar.gz -o helm.tar.gz
        tar -zxvf helm.tar.gz
        sudo mv linux-amd64/helm /usr/local/bin/helm
        rm -rf linux-amd64 helm.tar.gz
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install helm
    else
        log_warning "Please install Helm manually for Windows"
    fi

    log_success "Helm installed"
}

# Install Istioctl
install_istioctl() {
    log_info "Installing Istioctl..."

    if command -v istioctl >/dev/null 2>&1; then
        log_info "Istioctl is already installed"
        return
    fi

    # Download and install Istioctl
    curl -L https://istio.io/downloadIstio | sh -
    cd istio-*
    sudo mv bin/istioctl /usr/local/bin/
    cd ..
    rm -rf istio-*

    log_success "Istioctl installed"
}

# Setup local registry
setup_local_registry() {
    log_info "Setting up local Docker registry..."

    # Start local registry
    docker run -d -p 5000:5000 --name registry registry:2

    # Configure Minikube to use local registry
    minikube addons configure registry-creds
    minikube addons enable registry-creds

    log_success "Local Docker registry setup"
}

# Main setup function
main() {
    log_info "Setting up local infrastructure for JPMorgan Financial APIs production deployment"

    check_os
    install_docker
    start_docker
    install_kubectl
    install_minikube
    start_minikube
    install_helm
    install_istioctl
    setup_local_registry

    log_success "🎉 Local infrastructure setup completed!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Verify cluster: kubectl get nodes"
    log_info "2. Verify Docker: docker ps"
    log_info "3. Run deployment: ./deploy_production_complete.sh"
    log_info ""
    log_info "To stop the cluster: minikube stop"
    log_info "To delete the cluster: minikube delete"
}

# Handle command line arguments
case "${1:-}" in
    "docker")
        install_docker
        start_docker
        ;;
    "k8s")
        install_kubectl
        install_minikube
        start_minikube
        ;;
    "tools")
        install_helm
        install_istioctl
        ;;
    "registry")
        setup_local_registry
        ;;
    *)
        main
        ;;
esac
