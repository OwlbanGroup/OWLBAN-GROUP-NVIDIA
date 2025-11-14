#!/bin/bash

# JPMorgan Financial APIs - Rollback Script
# This script provides rollback functionality for deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="./backups"
ROLLBACK_TIMEOUT=300

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

rollback_kubernetes() {
    log_info "Rolling back Kubernetes deployment..."

    # Check if deployment exists
    if ! kubectl get deployment jpmorgan-api &> /dev/null; then
        log_error "Kubernetes deployment not found"
        exit 1
    fi

    # Get current revision
    local current_revision
    current_revision=$(kubectl rollout history deployment/jpmorgan-api -o jsonpath='{.metadata.generation}')

    log_info "Current deployment revision: ${current_revision}"

    # Perform rollback
    kubectl rollout undo deployment/jpmorgan-api

    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/jpmorgan-api --timeout=${ROLLBACK_TIMEOUT}s

    # Verify rollback
    local new_revision
    new_revision=$(kubectl rollout history deployment/jpmorgan-api -o jsonpath='{.metadata.generation}')

    if [ "${new_revision}" != "${current_revision}" ]; then
        log_success "Kubernetes rollback completed successfully"
    else
        log_error "Kubernetes rollback failed"
        exit 1
    fi
}

rollback_docker_compose() {
    log_info "Rolling back Docker Compose deployment..."

    local compose_file="docker-compose.prod.yml"
    local project_name="jpmorgan_prod"

    # Check if services are running
    if ! docker-compose -f ${compose_file} -p ${project_name} ps | grep -q "Up"; then
        log_warning "No running services found"
        return 0
    fi

    # Stop current services
    log_info "Stopping current services..."
    docker-compose -f ${compose_file} -p ${project_name} down

    # Find the most recent backup
    local latest_backup
    latest_backup=$(find ${BACKUP_DIR} -name "postgres_backup.sql" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -n "${latest_backup}" ]; then
        log_info "Restoring from backup: ${latest_backup}"

        # Start only PostgreSQL and Redis
        docker-compose -f ${compose_file} -p ${project_name} up -d postgres redis

        # Wait for PostgreSQL to be ready
        log_info "Waiting for PostgreSQL to be ready..."
        timeout=60
        while [ $timeout -gt 0 ]; do
            if docker-compose -f ${compose_file} -p ${project_name} exec -T postgres pg_isready -U telemetry_user -d telemetry_db > /dev/null 2>&1; then
                break
            fi
            sleep 2
            timeout=$((timeout - 2))
        done

        if [ $timeout -le 0 ]; then
            log_error "PostgreSQL failed to start"
            exit 1
        fi

        # Restore database
        log_info "Restoring database..."
        docker-compose -f ${compose_file} -p ${project_name} exec -T postgres psql -U telemetry_user -d telemetry_db < "${latest_backup}"

        log_success "Database restored from backup"
    else
        log_warning "No database backup found, starting with fresh database"
    fi

    # Start all services
    log_info "Starting all services..."
    docker-compose -f ${compose_file} -p ${project_name} up -d

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30

    log_success "Docker Compose rollback completed"
}

rollback_to_previous_version() {
    log_info "Rolling back to previous version..."

    # This would typically involve:
    # 1. Pulling previous Docker image tag
    # 2. Updating deployment manifests
    # 3. Applying changes

    log_warning "Automatic version rollback not implemented yet"
    log_info "Please manually specify the version to rollback to"
}

list_backups() {
    log_info "Available backups:"

    if [ -d "${BACKUP_DIR}" ]; then
        find ${BACKUP_DIR} -name "*.sql" -o -name "*.rdb" | sort -r | head -10
    else
        log_warning "No backup directory found"
    fi
}

cleanup_failed_deployment() {
    log_info "Cleaning up failed deployment..."

    # Remove failed pods
    kubectl delete pods --field-selector=status.phase=Failed -l app=jpmorgan-api --ignore-not-found=true

    # Remove failed deployments
    kubectl delete deployment jpmorgan-api --ignore-not-found=true

    # Clean up Docker containers
    docker container prune -f
    docker image prune -f

    log_success "Cleanup completed"
}

show_rollback_status() {
    log_info "Rollback Status:"

    echo ""
    echo "Kubernetes Deployments:"
    kubectl get deployments -l app=jpmorgan-api 2>/dev/null || echo "No Kubernetes deployments found"

    echo ""
    echo "Docker Compose Services:"
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.prod.yml -p jpmorgan_prod ps 2>/dev/null || echo "No Docker Compose services found"
    else
        echo "docker-compose not available"
    fi

    echo ""
    list_backups
}

# Main script
main() {
    echo "JPMorgan Financial APIs - Rollback Script"
    echo "========================================"

    # Detect deployment type
    local deployment_type=""

    if kubectl get deployment jpmorgan-api &> /dev/null; then
        deployment_type="kubernetes"
    elif docker-compose -f docker-compose.prod.yml -p jpmorgan_prod ps &> /dev/null && docker-compose -f docker-compose.prod.yml -p jpmorgan_prod ps | grep -q "Up"; then
        deployment_type="docker-compose"
    fi

    case "${1:-auto}" in
        "k8s"|"kubernetes")
            rollback_kubernetes
            ;;
        "docker"|"compose")
            rollback_docker_compose
            ;;
        "version")
            rollback_to_previous_version
            ;;
        "cleanup")
            cleanup_failed_deployment
            ;;
        "status")
            show_rollback_status
            ;;
        "auto")
            if [ -n "${deployment_type}" ]; then
                log_info "Detected ${deployment_type} deployment"
                case "${deployment_type}" in
                    "kubernetes")
                        rollback_kubernetes
                        ;;
                    "docker-compose")
                        rollback_docker_compose
                        ;;
                esac
            else
                log_error "No active deployment detected"
                echo "Please specify rollback type: k8s, docker, or version"
                exit 1
            fi
            ;;
        *)
            echo "Usage: $0 {auto|k8s|docker|version|cleanup|status}"
            echo ""
            echo "Commands:"
            echo "  auto     - Auto-detect deployment type and rollback"
            echo "  k8s      - Rollback Kubernetes deployment"
            echo "  docker   - Rollback Docker Compose deployment"
            echo "  version  - Rollback to previous version"
            echo "  cleanup  - Clean up failed deployment artifacts"
            echo "  status   - Show rollback status and available backups"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
