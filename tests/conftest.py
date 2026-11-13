"""Shared pytest configuration for the OWLBAN GROUP test suite."""

import os

# api_server raises at import time when API_PASSWORD is unset, which would
# abort collection for every module that imports it. Provide a dummy value
# for the test run; CI sets its own value and that one wins.
os.environ.setdefault("API_PASSWORD", "test-password")
