-- Initialize database with proper user and permissions
CREATE USER IF NOT EXISTS jpmorgan WITH PASSWORD 'secure_password_123';
GRANT ALL PRIVILEGES ON DATABASE jpmorgan_api TO jpmorgan;
ALTER USER jpmorgan CREATEDB;

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Audit logging table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    endpoint VARCHAR(500),
    ip_address INET,
    user_agent TEXT,
    request_data JSONB,
    response_status INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
