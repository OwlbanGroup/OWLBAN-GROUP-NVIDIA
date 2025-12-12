#!/bin/bash

# JPMorgan Financial APIs - Health Monitoring Script
# This script monitors the API health and sends alerts if needed

API_URL="https://api.equityshieldadvocates.com"
LOG_FILE="/opt/jpmorgan-api/logs/monitor.log"
ALERT_EMAIL="admin@equityshieldadvocates.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
    echo -e "$1"
}

check_health() {
    if curl -f -s --max-time 10 $API_URL/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_database() {
    # Check if database file exists and is accessible
    if [ -f "/opt/jpmorgan-api/data/jpmorgan_api.db" ]; then
        return 0
    else
        return 1
    fi
}

check_disk_space() {
    # Check if disk usage is above 90%
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 90 ]; then
        return 1
    else
        return 0
    fi
}

check_memory() {
    # Check if memory usage is above 90%
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ $MEM_USAGE -gt 90 ]; then
        return 1
    else
        return 0
    fi
}

send_alert() {
    local subject="$1"
    local message="$2"

    # Send email alert (requires mailutils or similar)
    if command -v mail &> /dev/null; then
        echo "$message" | mail -s "$subject" $ALERT_EMAIL
    fi

    log "${RED}ALERT: $subject - $message${NC}"
}

restart_service() {
    log "${YELLOW}Attempting to restart services...${NC}"

    cd /opt/jpmorgan-api
    docker-compose restart

    sleep 30

    if check_health; then
        log "${GREEN}Services restarted successfully${NC}"
    else
        send_alert "CRITICAL: Service Restart Failed" "Failed to restart JPMorgan API services after health check failure"
    fi
}

# Main monitoring loop
log "Starting JPMorgan API monitoring service..."

FAILED_CHECKS=0
MAX_FAILURES=3

while true; do
    ISSUES_FOUND=0

    # Health check
    if ! check_health; then
        log "${RED}Health check failed${NC}"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    else
        log "${GREEN}Health check passed${NC}"
        FAILED_CHECKS=0
    fi

    # Database check
    if ! check_database; then
        log "${RED}Database check failed${NC}"
        send_alert "WARNING: Database Issue" "Database file not found or inaccessible"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi

    # Disk space check
    if ! check_disk_space; then
        log "${RED}Disk space check failed${NC}"
        send_alert "WARNING: Low Disk Space" "Disk usage above 90%"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi

    # Memory check
    if ! check_memory; then
        log "${RED}Memory check failed${NC}"
        send_alert "WARNING: High Memory Usage" "Memory usage above 90%"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi

    # Take action if too many failures
    if [ $FAILED_CHECKS -ge $MAX_FAILURES ]; then
        send_alert "CRITICAL: Multiple Health Check Failures" "API has failed $FAILED_CHECKS consecutive health checks"
        restart_service
        FAILED_CHECKS=0
    fi

    # Log status
    if [ $ISSUES_FOUND -eq 0 ]; then
        log "${GREEN}All checks passed${NC}"
    else
        log "${YELLOW}$ISSUES_FOUND issues found${NC}"
    fi

    # Wait before next check
    sleep 30
done
