"""
OWLBAN GROUP AI Database Manager
Unified database interface for all AI systems with SQL and NoSQL support
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import os

# Optional database drivers
try:
    import pymongo
    from pymongo import MongoClient
    mongodb_available = True
except ImportError:
    mongodb_available = False

try:
    import psycopg2
    postgresql_available = True
except ImportError:
    postgresql_available = False

try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False

class DatabaseManager:
    """Unified database manager supporting multiple database types"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("DatabaseManager")
        self.config = config or self._default_config()
        self.connections = {}

        # Initialize databases
        self._init_sqlite()
        if mongodb_available:
            self._init_mongodb()
        if postgresql_available:
            self._init_postgresql()
        if redis_available:
            self._init_redis()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "sqlite": {
                "path": "owlban_ai.db"
            },
            "mongodb": {
                "host": "localhost",
                "port": 27017,
                "database": "owlban_ai"
            },
            "postgresql": {
                "host": "localhost",
                "port": 5432,
                "database": "owlban_ai",
                "user": "owlban",
                "password": "password"
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            }
        }

    def _init_sqlite(self):
        """Initialize SQLite database"""
        try:
            db_path = self.config["sqlite"]["path"]
            self.connections["sqlite"] = sqlite3.connect(db_path)
            self._create_sqlite_tables()
            self.logger.info("SQLite database initialized")
        except Exception as e:
            self.logger.error(f"SQLite initialization failed: {e}")

    def _create_sqlite_tables(self):
        """Create necessary SQLite tables"""
        cursor = self.connections["sqlite"].cursor()

        # AI predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                input_data TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Revenue optimization results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS revenue_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                profit REAL,
                parameters TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # System metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                value REAL,
                tags TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Quantum computations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quantum_computations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                algorithm TEXT,
                input_parameters TEXT,
                result TEXT,
                execution_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.connections["sqlite"].commit()

    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            config = self.config["mongodb"]
            client = MongoClient(config["host"], config["port"])
            self.connections["mongodb"] = client[config["database"]]
            self.logger.info("MongoDB connection initialized")
        except Exception as e:
            self.logger.error(f"MongoDB initialization failed: {e}")

    def _init_postgresql(self):
        """Initialize PostgreSQL connection"""
        try:
            config = self.config["postgresql"]
            conn_string = f"host={config['host']} port={config['port']} dbname={config['database']} user={config['user']} password={config['password']}"
            self.connections["postgresql"] = psycopg2.connect(conn_string)
            self._create_postgresql_tables()
            self.logger.info("PostgreSQL connection initialized")
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization failed: {e}")

    def _create_postgresql_tables(self):
        """Create PostgreSQL tables if they don't exist"""
        cursor = self.connections["postgresql"].cursor()

        # Similar table structure as SQLite but for PostgreSQL
        tables = [
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                input_data JSONB,
                prediction JSONB,
                confidence DOUBLE PRECISION,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS revenue_optimization (
                id SERIAL PRIMARY KEY,
                strategy VARCHAR(255),
                profit DOUBLE PRECISION,
                parameters JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(255),
                value DOUBLE PRECISION,
                tags JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]

        for table_sql in tables:
            try:
                cursor.execute(table_sql)
            except Exception as e:
                self.logger.warning(f"Table creation failed: {e}")

        self.connections["postgresql"].commit()
        cursor.close()

    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            config = self.config["redis"]
            self.connections["redis"] = redis.Redis(
                host=config["host"],
                port=config["port"],
                db=config["db"]
            )
            self.logger.info("Redis connection initialized")
        except Exception as e:
            self.logger.error(f"Redis initialization failed: {e}")

    # SQLite operations
    def save_prediction_sqlite(self, model_name: str, input_data: Dict, prediction: Any, confidence: float):
        """Save prediction to SQLite"""
        if "sqlite" not in self.connections:
            return False

        try:
            cursor = self.connections["sqlite"].cursor()
            cursor.execute(
                "INSERT INTO predictions (model_name, input_data, prediction, confidence) VALUES (?, ?, ?, ?)",
                (model_name, json.dumps(input_data), json.dumps(prediction), confidence)
            )
            self.connections["sqlite"].commit()
            return True
        except Exception as e:
            self.logger.error(f"SQLite save prediction failed: {e}")
            return False

    def get_predictions_sqlite(self, model_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get predictions from SQLite"""
        if "sqlite" not in self.connections:
            return []

        try:
            cursor = self.connections["sqlite"].cursor()
            if model_name:
                cursor.execute(
                    "SELECT * FROM predictions WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
                    (model_name, limit)
                )
            else:
                cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,))

            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON fields
                if result.get('input_data'):
                    result['input_data'] = json.loads(result['input_data'])
                if result.get('prediction'):
                    result['prediction'] = json.loads(result['prediction'])
                results.append(result)

            return results
        except Exception as e:
            self.logger.error(f"SQLite get predictions failed: {e}")
            return []

    # MongoDB operations
    def save_prediction_mongodb(self, model_name: str, input_data: Dict, prediction: Any, confidence: float):
        """Save prediction to MongoDB"""
        if "mongodb" not in self.connections:
            return False

        try:
            collection = self.connections["mongodb"].predictions
            doc = {
                "model_name": model_name,
                "input_data": input_data,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": datetime.utcnow()
            }
            collection.insert_one(doc)
            return True
        except Exception as e:
            self.logger.error(f"MongoDB save prediction failed: {e}")
            return False

    # Redis operations
    def cache_prediction_redis(self, key: str, prediction: Dict, ttl: int = 3600):
        """Cache prediction in Redis"""
        if "redis" not in self.connections:
            return False

        try:
            self.connections["redis"].setex(key, ttl, json.dumps(prediction))
            return True
        except Exception as e:
            self.logger.error(f"Redis cache prediction failed: {e}")
            return False

    def get_cached_prediction_redis(self, key: str) -> Optional[Dict]:
        """Get cached prediction from Redis"""
        if "redis" not in self.connections:
            return None

        try:
            data = self.connections["redis"].get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Redis get cached prediction failed: {e}")
            return None

    # Unified interface
    def save_prediction(self, model_name: str, input_data: Dict, prediction: Any, confidence: float):
        """Save prediction to all available databases"""
        results = []

        # Save to SQLite
        results.append(("sqlite", self.save_prediction_sqlite(model_name, input_data, prediction, confidence)))

        # Save to MongoDB if available
        if mongodb_available:
            results.append(("mongodb", self.save_prediction_mongodb(model_name, input_data, prediction, confidence)))

        # Cache in Redis if available
        if redis_available:
            cache_key = f"prediction:{model_name}:{hash(str(input_data))}"
            cache_data = {
                "model_name": model_name,
                "input_data": input_data,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(("redis", self.cache_prediction_redis(cache_key, cache_data)))

        return results

    def get_predictions(self, model_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get predictions from primary database (SQLite)"""
        return self.get_predictions_sqlite(model_name, limit)

    def save_revenue_result(self, strategy: str, profit: float, parameters: Dict):
        """Save revenue optimization result"""
        if "sqlite" not in self.connections:
            return False

        try:
            cursor = self.connections["sqlite"].cursor()
            cursor.execute(
                "INSERT INTO revenue_optimization (strategy, profit, parameters) VALUES (?, ?, ?)",
                (strategy, profit, json.dumps(parameters))
            )
            self.connections["sqlite"].commit()
            return True
        except Exception as e:
            self.logger.error(f"Save revenue result failed: {e}")
            return False

    def save_system_metric(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """Save system metric"""
        if "sqlite" not in self.connections:
            return False

        try:
            cursor = self.connections["sqlite"].cursor()
            cursor.execute(
                "INSERT INTO system_metrics (metric_name, value, tags) VALUES (?, ?, ?)",
                (metric_name, value, json.dumps(tags or {}))
            )
            self.connections["sqlite"].commit()
            return True
        except Exception as e:
            self.logger.error(f"Save system metric failed: {e}")
            return False

    def save_quantum_computation(self, algorithm: str, input_parameters: Dict, result: Any, execution_time: float):
        """Save quantum computation result"""
        if "sqlite" not in self.connections:
            return False

        try:
            cursor = self.connections["sqlite"].cursor()
            cursor.execute(
                "INSERT INTO quantum_computations (algorithm, input_parameters, result, execution_time) VALUES (?, ?, ?, ?)",
                (algorithm, json.dumps(input_parameters), json.dumps(result), execution_time)
            )
            self.connections["sqlite"].commit()
            return True
        except Exception as e:
            self.logger.error(f"Save quantum computation failed: {e}")
            return False

    def get_database_status(self) -> Dict[str, Any]:
        """Get status of all database connections"""
        status = {}
        for db_type, connection in self.connections.items():
            try:
                if db_type == "sqlite":
                    cursor = connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM predictions")
                    count = cursor.fetchone()[0]
                    status[db_type] = {"connected": True, "predictions_count": count}
                elif db_type == "mongodb":
                    status[db_type] = {"connected": True, "collections": connection.list_collection_names()}
                elif db_type == "postgresql":
                    cursor = connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM predictions")
                    count = cursor.fetchone()[0]
                    status[db_type] = {"connected": True, "predictions_count": count}
                    cursor.close()
                elif db_type == "redis":
                    status[db_type] = {"connected": connection.ping(), "db": connection.connection.db}
            except Exception as e:
                status[db_type] = {"connected": False, "error": str(e)}

        return status

    def close_all(self):
        """Close all database connections"""
        for db_type, connection in self.connections.items():
            try:
                if db_type in ["sqlite", "postgresql"]:
                    connection.close()
                elif db_type == "mongodb":
                    connection.client.close()
                elif db_type == "redis":
                    connection.close()
                self.logger.info(f"Closed {db_type} connection")
            except Exception as e:
                self.logger.error(f"Error closing {db_type}: {e}")
