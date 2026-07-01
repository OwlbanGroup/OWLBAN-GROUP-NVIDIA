"""
OWLBAN GROUP AI Database Manager
Unified database interface for all AI systems with SQL and NoSQL support
"""

import sqlite3
import json
import logging
import importlib
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime, timezone

# Optional database drivers (import dynamically to avoid static-type issues)
MongoClient = None
psycopg2 = None
redis = None
try:
    MongoClient = importlib.import_module("pymongo").MongoClient
    MONGODB_AVAILABLE = True
except Exception:
    MONGODB_AVAILABLE = False

try:
    psycopg2 = importlib.import_module("psycopg2")
    POSTGRESQL_AVAILABLE = True
except Exception:
    POSTGRESQL_AVAILABLE = False

try:
    redis = importlib.import_module("redis")
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

class DatabaseManager:
    """Unified database manager supporting multiple database types"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("DatabaseManager")
        self.config = config or self._default_config()
        self.connections: Dict[str, Any] = {}

        # Initialize databases
        self._init_sqlite()
        if MONGODB_AVAILABLE:
            self._init_mongodb()
        if POSTGRESQL_AVAILABLE:
            self._init_postgresql()
        if REDIS_AVAILABLE:
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
        except sqlite3.Error as e:
            self.logger.error("SQLite initialization failed: %s", e)

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
            self.logger.error("MongoDB initialization failed: %s", e)

    def _init_postgresql(self):
        """Initialize PostgreSQL connection"""
        try:
            config = self.config["postgresql"]
            conn_string = (
                "host=%s port=%s dbname=%s user=%s password=%s"
                % (config["host"], config["port"], config["database"], config["user"], config["password"])
            )
            self.connections["postgresql"] = psycopg2.connect(conn_string)
            self._create_postgresql_tables()
            self.logger.info("PostgreSQL connection initialized")
        except Exception as e:
            self.logger.error("PostgreSQL initialization failed: %s", e)

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
            self.logger.error("Redis initialization failed: %s", e)

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
        except sqlite3.Error as e:
            self.logger.error("SQLite save prediction failed: %s", e)
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
        except sqlite3.Error as e:
            self.logger.error("SQLite get predictions failed: %s", e)
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
                "timestamp": datetime.now(timezone.utc)
            }
            collection.insert_one(doc)
            return True
        except Exception as e:
            self.logger.error("MongoDB save prediction failed: %s", e)
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
            self.logger.error("Redis cache prediction failed: %s", e)
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
            self.logger.error("Redis get cached prediction failed: %s", e)
            return None

    # Unified interface
    def save_prediction(self, model_name: str, input_data: Dict, prediction: Any, confidence: float):
        """Save prediction to all available databases"""
        results = []

        # Save to SQLite
        results.append(("sqlite", self.save_prediction_sqlite(model_name, input_data, prediction, confidence)))

        # Save to MongoDB if available
        if MONGODB_AVAILABLE:
            results.append(("mongodb", self.save_prediction_mongodb(model_name, input_data, prediction, confidence)))

        # Cache in Redis if available
        if REDIS_AVAILABLE:
            cache_key = "prediction:%s:%s" % (model_name, hash(str(input_data)))
            cache_data = {
                "model_name": model_name,
                "input_data": input_data,
                "prediction": prediction,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
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
        except sqlite3.Error as e:
            self.logger.error("Save revenue result failed: %s", e)
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
        except sqlite3.Error as e:
            self.logger.error("Save system metric failed: %s", e)
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
        except sqlite3.Error as e:
            self.logger.error("Save quantum computation failed: %s", e)
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
                    status[db_type] = {"connected": connection.ping(), "db": getattr(connection, 'connection', {}).get('db', None)}
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
                    # pymongo database object holds a client attribute
                    client = getattr(connection, 'client', None)
                    if client:
                        client.close()
                elif db_type == "redis":
                    connection.close()
                self.logger.info("Closed %s connection", db_type)
            except Exception as e:
                self.logger.error("Error closing %s: %s", db_type, e)
