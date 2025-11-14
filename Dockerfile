# Multi-stage Dockerfile for JPMorgan Financial APIs
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements_new.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_new.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs backups temp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "production_server.py"]

# Production stage
FROM base as production

# Switch back to root to install packages
USER root

# Install production dependencies
RUN pip install --no-cache-dir waitress gunicorn

# Set production environment
ENV FLASK_ENV=production \
    TESTING=false \
    SECRET_KEY=production-secret-key-change-this

# Switch back to appuser
USER appuser

# Use production server
CMD ["python", "production_server.py"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy

# Set development environment
ENV FLASK_ENV=development \
    TESTING=true

# Use development server
CMD ["python", "app_final.py"]

# Testing stage
FROM base as testing

# Install testing dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-xdist

# Copy test files
COPY test_*.py ./

# Run tests
CMD ["pytest", "--cov=.", "--cov-report=html", "--cov-report=term"]

# GPU-enabled stage (optional)
FROM python:3.11-slim as gpu

# Install CUDA and GPU dependencies
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Install GPU-specific packages
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir cuml-cu11 --extra-index-url https://pypi.ngc.nvidia.com

# Copy application code
COPY . .

# Set GPU environment
ENV GPU_ENABLED=true \
    CUDA_VISIBLE_DEVICES=0

CMD ["python", "production_server.py"]
