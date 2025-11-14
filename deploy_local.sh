#!/bin/bash

# Local Development Deployment Script for JPMorgan Financial APIs
# This script sets up the application for local development

set -e

echo "🚀 Starting Local Development Deployment for JPMorgan Financial APIs..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(printf '%s\n' "$PYTHON_VERSION" "3.9" | sort -V | head -n1)" != "3.9" ]]; then
    echo "❌ Python 3.9+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
if [ -f "requirements_new.txt" ]; then
    pip install -r requirements_new.txt
else
    pip install -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p .pytest_cache

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📋 Setting up environment configuration..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before running the application"
fi

# Run database migrations/initialization if needed
echo "🗄️ Initializing database..."
python3 -c "
from config import config
from src.telemetry_handler import telemetry_handler
print('Database initialized successfully')
"

# Run tests to ensure everything works
echo "🧪 Running tests..."
python3 -m pytest test_additional_endpoints.py -v --tb=short

# Start the application
echo "🎯 Starting the application..."
echo "📊 Application will be available at: http://localhost:5000"
echo "🏥 Health check endpoint: http://localhost:5000/health"
echo "📖 API documentation: http://localhost:5000/swagger"
echo ""

# Install NVIDIA NGC CLI if not installed
if ! command -v ngc &> /dev/null; then
    echo "📥 Installing NVIDIA NGC CLI..."
    curl -L https://ngc.nvidia.com/downloads/ngccli_linux.zip -o ngccli_linux.zip
    unzip ngccli_linux.zip -d ngccli
    sudo cp ngccli/ngc /usr/local/bin/ngc
    rm -rf ngccli ngccli_linux.zip
fi

# Configure NVIDIA NGC API key
if [ -n "$NGC_API_KEY" ]; then
    echo "🔑 Configuring NVIDIA NGC API key..."
    ngc config set apiKey $NGC_API_KEY
else
    echo "⚠️ NGC_API_KEY environment variable not set. Please set it to use NVIDIA NGC services"
fi

echo ""
echo "To stop the application, press Ctrl+C"
echo ""

python3 app.py
