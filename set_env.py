#!/usr/bin/env python
"""Set environment variables for OWLBAN-GROUP-NVIDIA project"""
import os
import sys

# Set API_PASSWORD
os.environ['API_PASSWORD'] = 'testpassword123'

# Verify it's set
print(f"API_PASSWORD set to: {os.environ.get('API_PASSWORD')}")

# Also set other useful env vars
os.environ.setdefault('ENVIRONMENT', 'development')
os.environ.setdefault('LOG_LEVEL', 'INFO')

print("Environment variables configured.")