"""
Database manager module - provides get_session function for tests
"""
from unittest.mock import Mock

def get_session():
    """Context manager mock for SQLAlchemy session"""
    class MockSession:
        def __enter__(self):
            return Mock()
        def __exit__(self, *args):
            pass
    return MockSession()

