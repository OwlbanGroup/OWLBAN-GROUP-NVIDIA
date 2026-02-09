#!/usr/bin/env python3
"""
Database initialization script for JPMorgan Financial APIs
Creates all required tables, indexes, and initial data
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3
from datetime import datetime, timezone
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection based on environment"""
    db_type = os.getenv('DATABASE_TYPE', 'sqlite')

    if db_type == 'postgresql':
        try:
            conn = psycopg2.connect(
                host=os.getenv('DATABASE_HOST', 'localhost'),
                port=int(os.getenv('DATABASE_PORT', 5432)),
                database=os.getenv('DATABASE_NAME', 'jpmorgan_financial'),
                user=os.getenv('DATABASE_USER', 'jpmorgan_user'),
                password=os.getenv('DATABASE_PASSWORD', 'jpmorgan_password_2024'),
                sslmode=os.getenv('DATABASE_SSL_MODE', 'require')
            )
            logger.info("Connected to PostgreSQL database")
            return conn, 'postgresql'
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            sys.exit(1)
    else:
        # SQLite fallback
        db_path = os.getenv('DATABASE_URL', 'sqlite:///jpmorgan_api.db').replace('sqlite:///', '')
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to SQLite database: {db_path}")
        return conn, 'sqlite'

def execute_sql_file(conn, file_path, db_type):
    """Execute SQL commands from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Split SQL commands (basic approach - may need refinement for complex scripts)
        commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip()]

        cursor = conn.cursor()
        for command in commands:
            if command:
                try:
                    cursor.execute(command)
                    logger.info(f"Executed: {command[:50]}...")
                except Exception as e:
                    logger.warning(f"Failed to execute command: {command[:50]}... Error: {e}")

        conn.commit()
        cursor.close()
        logger.info(f"Successfully executed SQL file: {file_path}")

    except FileNotFoundError:
        logger.error(f"SQL file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error executing SQL file {file_path}: {e}")

def create_tables(conn, db_type):
    """Create all required database tables"""

    # Users table
    users_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(255) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE,
        password_hash VARCHAR(255) NOT NULL,
        role VARCHAR(50) DEFAULT 'USER',
        business_id INTEGER,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        last_login_at TIMESTAMP WITH TIME ZONE,
        login_attempts INTEGER DEFAULT 0,
        locked_until TIMESTAMP WITH TIME ZONE
    );
    """

    # Businesses table
    businesses_sql = """
    CREATE TABLE IF NOT EXISTS businesses (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        type VARCHAR(100),
        registration_number VARCHAR(100),
        address TEXT,
        contact_info JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Assets table
    assets_sql = """
    CREATE TABLE IF NOT EXISTS assets (
        id SERIAL PRIMARY KEY,
        business_id INTEGER REFERENCES businesses(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        type VARCHAR(100),
        value DECIMAL(15,2),
        acquisition_date DATE,
        ownership_percentage DECIMAL(5,2) DEFAULT 100.00,
        description TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Revenue transactions table
    revenue_transactions_sql = """
    CREATE TABLE IF NOT EXISTS revenue_transactions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        revenue_type VARCHAR(50) NOT NULL,
        amount DECIMAL(15,2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'USD',
        description TEXT,
        merchant_name VARCHAR(255),
        category VARCHAR(100),
        payment_method VARCHAR(50),
        business_id INTEGER REFERENCES businesses(id),
        external_reference VARCHAR(255),
        metadata JSONB,
        status VARCHAR(50) DEFAULT 'pending',
        processed_at TIMESTAMP WITH TIME ZONE,
        settlement_date TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Payments table
    payments_sql = """
    CREATE TABLE IF NOT EXISTS payments (
        id SERIAL PRIMARY KEY,
        transaction_id INTEGER REFERENCES revenue_transactions(id),
        amount DECIMAL(15,2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'USD',
        payment_type VARCHAR(50),
        payment_status VARCHAR(50) DEFAULT 'pending',
        payment_method VARCHAR(50),
        external_payment_id VARCHAR(255),
        metadata JSONB,
        processed_at TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Audit logs table
    audit_logs_sql = """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER,
        username VARCHAR(255),
        action VARCHAR(255) NOT NULL,
        resource_type VARCHAR(100),
        resource_id VARCHAR(255),
        status_code INTEGER,
        request_data JSONB,
        response_data JSONB,
        severity VARCHAR(50) DEFAULT 'info',
        category VARCHAR(100),
        compliance_tags TEXT[],
        ip_address INET,
        user_agent TEXT,
        session_id VARCHAR(255),
        request_id VARCHAR(255),
        response_time_ms INTEGER,
        error_message TEXT,
        hash_chain VARCHAR(64),
        previous_hash VARCHAR(64)
    );
    """

    # Audit alerts table
    audit_alerts_sql = """
    CREATE TABLE IF NOT EXISTS audit_alerts (
        id SERIAL PRIMARY KEY,
        alert_type VARCHAR(100) NOT NULL,
        severity VARCHAR(50) DEFAULT 'medium',
        message TEXT NOT NULL,
        details JSONB,
        acknowledged BOOLEAN DEFAULT FALSE,
        acknowledged_by INTEGER,
        acknowledged_at TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP WITH TIME ZONE
    );
    """

    # Telemetry events table
    telemetry_events_sql = """
    CREATE TABLE IF NOT EXISTS telemetry_events (
        id SERIAL PRIMARY KEY,
        event_name VARCHAR(500) NOT NULL,
        event_version VARCHAR(50),
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        user_id VARCHAR(255),
        device_info JSONB,
        event_data JSONB,
        processed BOOLEAN DEFAULT FALSE,
        anomaly_score DECIMAL(5,4),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Indexes for performance
    indexes_sql = """
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_businesses_name ON businesses(name);
    CREATE INDEX IF NOT EXISTS idx_assets_business_id ON assets(business_id);
    CREATE INDEX IF NOT EXISTS idx_revenue_transactions_user_id ON revenue_transactions(user_id);
    CREATE INDEX IF NOT EXISTS idx_revenue_transactions_status ON revenue_transactions(status);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
    CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
    CREATE INDEX IF NOT EXISTS idx_telemetry_events_timestamp ON telemetry_events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_telemetry_events_user_id ON telemetry_events(user_id);
    """

    tables = [
        ("users", users_sql),
        ("businesses", businesses_sql),
        ("assets", assets_sql),
        ("revenue_transactions", revenue_transactions_sql),
        ("payments", payments_sql),
        ("audit_logs", audit_logs_sql),
        ("audit_alerts", audit_alerts_sql),
        ("telemetry_events", telemetry_events_sql)
    ]

    cursor = conn.cursor()

    for table_name, sql in tables:
        try:
            cursor.execute(sql)
            logger.info(f"Created table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")

    # Create indexes
    try:
        cursor.execute(indexes_sql)
        logger.info("Created database indexes")
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")

    conn.commit()
    cursor.close()

def insert_initial_data(conn, db_type):
    """Insert initial test data"""

    # Insert test user
    test_user_sql = """
    INSERT INTO users (username, email, password_hash, role)
    VALUES ('admin', 'admin@jpmorgan.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fMmiC3c6e', 'ADMIN')
    ON CONFLICT (username) DO NOTHING;
    """

    # Insert sample business
    sample_business_sql = """
    INSERT INTO businesses (name, type, registration_number, address, contact_info)
    VALUES ('JPMorgan Chase & Co.', 'corporation', '001-00001', '270 Park Avenue, New York, NY 10017', '{"email": "contact@jpmorgan.com", "phone": "+1-212-270-6000"}')
    ON CONFLICT DO NOTHING;
    """

    cursor = conn.cursor()

    try:
        cursor.execute(test_user_sql)
        logger.info("Inserted test user")
    except Exception as e:
        logger.warning(f"Failed to insert test user: {e}")

    try:
        cursor.execute(sample_business_sql)
        logger.info("Inserted sample business")
    except Exception as e:
        logger.warning(f"Failed to insert sample business: {e}")

    conn.commit()
    cursor.close()

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")

    # Check if we should skip initialization
    if os.getenv('SKIP_DB_INIT', '').lower() == 'true':
        logger.info("Database initialization skipped by environment variable")
        return

    conn, db_type = get_db_connection()

    try:
        # Create tables
        create_tables(conn, db_type)

        # Execute schema file if it exists
        schema_file = 'jpmorgan_database_schema.sql'
        if os.path.exists(schema_file):
            execute_sql_file(conn, schema_file, db_type)

        # Insert initial data
        insert_initial_data(conn, db_type)

        logger.info("Database initialization completed successfully!")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
