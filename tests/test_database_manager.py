"""
Comprehensive tests for the OWLBAN GROUP Database Manager.
Tests SQLite operations, connection management, and data integrity.
"""

import sys
import os
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_manager import DatabaseManager


@pytest.fixture
def temp_db():
    """Create a DatabaseManager with a temporary SQLite database."""
    db_path = os.path.join(tempfile.gettempdir(), f"test_owlban_{id(temp_db)}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = DatabaseManager(config={"sqlite": {"path": db_path}})
    yield db
    try:
        db.close_all()
    except Exception:
        pass
    if os.path.exists(db_path):
        os.remove(db_path)


class TestDatabaseInitialization:
    """Test database manager initialization and connection status."""

    def test_sqlite_connection(self, temp_db):
        """SQLite database should be connected."""
        status = temp_db.get_database_status()
        assert "sqlite" in status
        assert status["sqlite"]["connected"] is True

    def test_default_config_has_required_keys(self):
        """Default config should contain all required database sections."""
        db = DatabaseManager.__new__(DatabaseManager)
        config = db._default_config()
        assert "sqlite" in config
        assert "mongodb" in config
        assert "postgresql" in config
        assert "redis" in config
        assert "path" in config["sqlite"]


class TestPredictionOperations:
    """Test CRUD operations for AI predictions."""

    def test_save_and_get_prediction(self, temp_db):
        """Should save a prediction and retrieve it."""
        result = temp_db.save_prediction(
            model_name="test_model",
            input_data='{"feature": 1.0}',
            prediction='{"result": 0.5}',
            confidence=0.85,
        )
        # save_prediction returns a list of (db_type, success) tuples
        assert isinstance(result, list)
        assert any(r[0] == "sqlite" and r[1] is True for r in result)

        predictions = temp_db.get_predictions(limit=10)
        assert len(predictions) == 1
        assert predictions[0]["model_name"] == "test_model"
        assert predictions[0]["confidence"] == 0.85

    def test_save_multiple_predictions(self, temp_db):
        """Should save and retrieve multiple predictions."""
        for i in range(5):
            temp_db.save_prediction(
                model_name=f"model_{i}",
                input_data=f'{{"x": {i}}}',
                prediction=f'{{"y": {i * 2}}}',
                confidence=0.5 + i * 0.1,
            )

        predictions = temp_db.get_predictions(limit=10)
        assert len(predictions) == 5

    def test_get_predictions_limit(self, temp_db):
        """Should respect the limit parameter."""
        for i in range(10):
            temp_db.save_prediction(
                model_name="model",
                input_data='{}',
                prediction='{}',
                confidence=0.5,
            )

        predictions = temp_db.get_predictions(limit=3)
        assert len(predictions) == 3


class TestSystemMetrics:
    """Test system metrics operations."""

    def test_save_system_metric(self, temp_db):
        """Should save system metrics."""
        result = temp_db.save_system_metric("cpu_usage", 42.5, {"host": "server1"})
        assert result is True

    def test_save_multiple_metrics(self, temp_db):
        """Should save multiple system metrics."""
        for i in range(5):
            result = temp_db.save_system_metric(f"metric_{i}", float(i), {})
            assert result is True


class TestQuantumComputations:
    """Test quantum computation storage."""

    def test_save_quantum_computation(self, temp_db):
        """Should save quantum computation results."""
        result = temp_db.save_quantum_computation(
            algorithm="qaoa",
            input_parameters={"qubits": 4, "layers": 2},
            result={"energy": -1.5},
            execution_time=0.123,
        )
        assert result is True

    def test_get_database_status(self, temp_db):
        """Database status should report SQLite as connected."""
        status = temp_db.get_database_status()
        assert status["sqlite"]["connected"] is True
        assert "predictions_count" in status["sqlite"]


class TestCloseAll:
    """Test connection cleanup."""

    def test_close_all(self, temp_db):
        """Should close all connections without error."""
        temp_db.close_all()
