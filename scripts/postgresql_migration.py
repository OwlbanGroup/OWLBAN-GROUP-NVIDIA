#!/usr/bin/env python3
"""
PostgreSQL Migration Script for JPMorgan Financial APIs
Migrates from SQLite to PostgreSQL with proper schema conversion
"""
import sqlite3
import psycopg2
import psycopg2.extras
import json
from datetime import datetime
from config import Config

class PostgreSQLMigrator:
    """Handles migration from SQLite to PostgreSQL"""

    def __init__(self):
        self.sqlite_conn = None
        self.postgres_conn = None
        self.config = Config()

    def connect_sqlite(self):
        """Connect to SQLite database"""
        self.sqlite_conn = sqlite3.connect('telemetry.db')
        self.sqlite_conn.row_factory = sqlite3.Row

    def connect_postgresql(self):
        """Connect to PostgreSQL database"""
        self.postgres_conn = psycopg2.connect(
            host=self.config.DATABASE_HOST,
            port=self.config.DATABASE_PORT,
            database=self.config.DATABASE_NAME,
            user=self.config.DATABASE_USER,
            password=self.config.DATABASE_PASSWORD,
            sslmode=self.config.DATABASE_SSL_MODE
        )
        self.postgres_conn.autocommit = False

    def create_postgresql_schema(self):
        """Create PostgreSQL schema"""
        schema_sql = """
        -- Create telemetry_events table
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id SERIAL PRIMARY KEY,
            event_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP WITH TIME ZONE,
            status VARCHAR(50) DEFAULT 'pending'
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_telemetry_events_created_at ON telemetry_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_telemetry_events_status ON telemetry_events(status);
        CREATE INDEX IF NOT EXISTS idx_telemetry_events_event_data ON telemetry_events USING GIN(event_data);

        -- Create users table
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            email VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP WITH TIME ZONE
        );

        -- Create businesses table
        CREATE TABLE IF NOT EXISTS businesses (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            industry VARCHAR(255),
            location VARCHAR(255),
            contact_email VARCHAR(255),
            website VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Create assets table
        CREATE TABLE IF NOT EXISTS assets (
            id SERIAL PRIMARY KEY,
            business_id INTEGER REFERENCES businesses(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            asset_type VARCHAR(100),
            value DECIMAL(15,2),
            location VARCHAR(255),
            status VARCHAR(50) DEFAULT 'Active',
            purchase_date DATE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Create ML models table
        CREATE TABLE IF NOT EXISTS ml_models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            model_type VARCHAR(100),
            model_data BYTEA,
            parameters JSONB,
            accuracy DECIMAL(5,4),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Create API logs table
        CREATE TABLE IF NOT EXISTS api_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            method VARCHAR(10),
            endpoint VARCHAR(500),
            status_code INTEGER,
            response_time DECIMAL(10,4),
            user_id INTEGER,
            ip_address INET,
            user_agent TEXT
        );

        -- Create indexes for API logs
        CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp ON api_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs(endpoint);
        CREATE INDEX IF NOT EXISTS idx_api_logs_status_code ON api_logs(status_code);
        """

        with self.postgres_conn.cursor() as cursor:
            cursor.execute(schema_sql)
        self.postgres_conn.commit()

    def migrate_telemetry_data(self):
        """Migrate telemetry data from SQLite to PostgreSQL"""
        print("Migrating telemetry data...")

        # Get data from SQLite
        sqlite_cursor = self.sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT * FROM telemetry_events")
        rows = sqlite_cursor.fetchall()

        # Insert into PostgreSQL
        postgres_cursor = self.postgres_conn.cursor()

        for row in rows:
            # Convert SQLite row to dict
            row_dict = dict(row)

            # Handle JSON data conversion
            if 'event_data' in row_dict:
                # SQLite stores JSON as string, PostgreSQL uses JSONB
                if isinstance(row_dict['event_data'], str):
                    row_dict['event_data'] = json.loads(row_dict['event_data'])

            # Insert into PostgreSQL
            postgres_cursor.execute("""
                INSERT INTO telemetry_events (id, event_data, created_at, processed_at, status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (
                row_dict.get('id'),
                json.dumps(row_dict.get('event_data', {})),
                row_dict.get('created_at'),
                row_dict.get('processed_at'),
                row_dict.get('status', 'pending')
            ))

        self.postgres_conn.commit()
        print(f"Migrated {len(rows)} telemetry events")

    def migrate_user_data(self):
        """Migrate user data if exists"""
        print("Checking for user data...")

        try:
            sqlite_cursor = self.sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            if sqlite_cursor.fetchone():
                sqlite_cursor.execute("SELECT * FROM users")
                rows = sqlite_cursor.fetchall()

                postgres_cursor = self.postgres_conn.cursor()
                for row in rows:
                    row_dict = dict(row)
                    postgres_cursor.execute("""
                        INSERT INTO users (id, username, password_hash, email, created_at, last_login)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (username) DO NOTHING
                    """, (
                        row_dict.get('id'),
                        row_dict.get('username'),
                        row_dict.get('password_hash'),
                        row_dict.get('email'),
                        row_dict.get('created_at'),
                        row_dict.get('last_login')
                    ))

                self.postgres_conn.commit()
                print(f"Migrated {len(rows)} users")
            else:
                print("No users table found in SQLite")
        except Exception as e:
            print(f"User migration skipped: {e}")

    def validate_migration(self):
        """Validate that migration was successful"""
        print("Validating migration...")

        # Check row counts
        sqlite_cursor = self.sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT COUNT(*) FROM telemetry_events")
        sqlite_count = sqlite_cursor.fetchone()[0]

        postgres_cursor = self.postgres_conn.cursor()
        postgres_cursor.execute("SELECT COUNT(*) FROM telemetry_events")
        postgres_count = postgres_cursor.fetchone()[0]

        if sqlite_count == postgres_count:
            print(f"✅ Migration successful: {postgres_count} records migrated")
            return True
        else:
            print(f"❌ Migration validation failed: SQLite={sqlite_count}, PostgreSQL={postgres_count}")
            return False

    def run_migration(self):
        """Run the complete migration process"""
        print("🚀 Starting PostgreSQL Migration")
        print("=" * 50)

        try:
            # Connect to databases
            print("Connecting to databases...")
            self.connect_sqlite()
            self.connect_postgresql()

            # Create PostgreSQL schema
            print("Creating PostgreSQL schema...")
            self.create_postgresql_schema()

            # Migrate data
            self.migrate_telemetry_data()
            self.migrate_user_data()

            # Validate migration
            success = self.validate_migration()

            if success:
                print("\n✅ Migration completed successfully!")
                print("You can now update your DATABASE_URL to use PostgreSQL")
                return True
            else:
                print("\n❌ Migration validation failed!")
                self.postgres_conn.rollback()
                return False

        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            if self.postgres_conn:
                self.postgres_conn.rollback()
            return False

        finally:
            # Close connections
            if self.sqlite_conn:
                self.sqlite_conn.close()
            if self.postgres_conn:
                self.postgres_conn.close()

def main():
    """Main migration function"""
    migrator = PostgreSQLMigrator()
    success = migrator.run_migration()

    if success:
        print("\nNext steps:")
        print("1. Update your environment variables:")
        print("   DATABASE_TYPE=postgresql")
        print("   DATABASE_HOST=your_postgres_host")
        print("   DATABASE_USER=your_username")
        print("   DATABASE_PASSWORD=your_password")
        print("2. Restart the application")
        print("3. Test that everything works with PostgreSQL")
        print("4. Optionally, backup and remove the SQLite file")

    exit(0 if success else 1)

if __name__ == "__main__":
    main()
