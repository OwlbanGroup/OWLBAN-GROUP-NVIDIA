-- JPMorgan Financial APIs - Database Initialization Script
-- This script creates the necessary database schema and tables

-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS jpmorgan_financial_apis;
\c jpmorgan_financial_apis;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_buffercache";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS api;
CREATE SCHEMA IF NOT EXISTS telemetry;
CREATE SCHEMA IF NOT EXISTS audit;

-- Create telemetry tables
CREATE TABLE IF NOT EXISTS telemetry.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(100) NOT NULL,
    source VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS telemetry.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC NOT NULL,
    labels JSONB DEFAULT '{}',
    tags JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS telemetry.anomalies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    anomaly_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    description TEXT,
    data JSONB NOT NULL,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create API audit tables
CREATE TABLE IF NOT EXISTS audit.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    method VARCHAR(10) NOT NULL,
    path VARCHAR(500) NOT NULL,
    query_params JSONB DEFAULT '{}',
    request_body JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    user_agent TEXT,
    ip_address INET,
    headers JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_token VARCHAR(500) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Create API configuration tables
CREATE TABLE IF NOT EXISTS api.endpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    path VARCHAR(500) NOT NULL UNIQUE,
    method VARCHAR(10) NOT NULL,
    description TEXT,
    version VARCHAR(20) DEFAULT 'v1',
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit INTEGER DEFAULT 1000,
    timeout_seconds INTEGER DEFAULT 30,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(500) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_by VARCHAR(255),
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,
    permissions JSONB DEFAULT '[]',
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_telemetry_events_timestamp ON telemetry.events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_type ON telemetry.events (event_type);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_processed ON telemetry.events (processed) WHERE processed = FALSE;
CREATE INDEX IF NOT EXISTS idx_telemetry_metrics_timestamp ON telemetry.metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_metrics_name ON telemetry.metrics (metric_name);
CREATE INDEX IF NOT EXISTS idx_telemetry_anomalies_timestamp ON telemetry.anomalies (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_anomalies_severity ON telemetry.anomalies (severity);

CREATE INDEX IF NOT EXISTS idx_audit_requests_timestamp ON audit.api_requests (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_requests_user ON audit.api_requests (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_requests_path ON audit.api_requests (path);
CREATE INDEX IF NOT EXISTS idx_audit_sessions_user ON audit.user_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_sessions_token ON audit.user_sessions (session_token);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api.api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api.api_keys (is_active) WHERE is_active = TRUE;

-- Create partitions for telemetry tables (by month)
CREATE TABLE IF NOT EXISTS telemetry.events_y2024m01 PARTITION OF telemetry.events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS telemetry.events_y2024m02 PARTITION OF telemetry.events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Create default partition
CREATE TABLE IF NOT EXISTS telemetry.events_default PARTITION OF telemetry.events DEFAULT;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_telemetry_events_updated_at BEFORE UPDATE ON telemetry.events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_endpoints_updated_at BEFORE UPDATE ON api.endpoints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api.api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for monitoring
CREATE OR REPLACE VIEW telemetry.hourly_metrics AS
SELECT
    date_trunc('hour', timestamp) AS hour,
    metric_name,
    AVG(metric_value) AS avg_value,
    MIN(metric_value) AS min_value,
    MAX(metric_value) AS max_value,
    COUNT(*) AS count
FROM telemetry.metrics
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', timestamp), metric_name
ORDER BY hour DESC, metric_name;

CREATE OR REPLACE VIEW audit.daily_api_usage AS
SELECT
    DATE(timestamp) AS date,
    COUNT(*) AS total_requests,
    COUNT(DISTINCT user_id) AS unique_users,
    AVG(response_time_ms) AS avg_response_time,
    COUNT(CASE WHEN response_status >= 400 THEN 1 END) AS error_count
FROM audit.api_requests
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Insert default API endpoints
INSERT INTO api.endpoints (name, path, method, description) VALUES
('health_check', '/health', 'GET', 'Health check endpoint'),
('api_status', '/api/v1/status', 'GET', 'API status information'),
('telemetry_submit', '/api/v1/telemetry', 'POST', 'Submit telemetry data'),
('metrics_get', '/metrics', 'GET', 'Prometheus metrics endpoint')
ON CONFLICT (name) DO NOTHING;

-- Create roles and permissions
CREATE ROLE readonly_user;
GRANT CONNECT ON DATABASE jpmorgan_financial_apis TO readonly_user;
GRANT USAGE ON SCHEMA telemetry TO readonly_user;
GRANT USAGE ON SCHEMA audit TO readonly_user;
GRANT USAGE ON SCHEMA api TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA telemetry TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA audit TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA api TO readonly_user;

CREATE ROLE api_user;
GRANT CONNECT ON DATABASE jpmorgan_financial_apis TO api_user;
GRANT USAGE ON SCHEMA telemetry TO api_user;
GRANT USAGE ON SCHEMA audit TO api_user;
GRANT USAGE ON SCHEMA api TO api_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA telemetry TO api_user;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO api_user;
GRANT SELECT ON ALL TABLES IN SCHEMA api TO api_user;

-- Create admin user (password should be changed in production)
-- CREATE USER jpmorgan_admin WITH PASSWORD 'CHANGE_THIS_PASSWORD';
-- GRANT ALL PRIVILEGES ON DATABASE jpmorgan_financial_apis TO jpmorgan_admin;
-- GRANT ALL ON SCHEMA telemetry, audit, api TO jpmorgan_admin;

-- Enable Row Level Security (RLS) where appropriate
ALTER TABLE audit.api_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit.user_sessions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY api_requests_user_policy ON audit.api_requests
    FOR SELECT USING (user_id = current_user OR current_user = 'jpmorgan_admin');

CREATE POLICY user_sessions_user_policy ON audit.user_sessions
    FOR SELECT USING (user_id = current_user OR current_user = 'jpmorgan_admin');

-- Create materialized view for performance analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry.performance_summary AS
SELECT
    DATE_TRUNC('day', timestamp) AS date,
    COUNT(*) AS total_events,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) AS avg_processing_time,
    COUNT(CASE WHEN processed = TRUE THEN 1 END) AS processed_events,
    COUNT(CASE WHEN processed = FALSE THEN 1 END) AS pending_events
FROM telemetry.events
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY date DESC;

-- Create refresh function for materialized view
CREATE OR REPLACE FUNCTION refresh_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY telemetry.performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Create a cron job to refresh the materialized view (requires pg_cron extension)
-- SELECT cron.schedule('refresh-performance-summary', '0 */6 * * *', 'SELECT refresh_performance_summary();');

-- Log initialization completion
DO $$
BEGIN
    RAISE NOTICE 'JPMorgan Financial APIs database initialization completed successfully';
END $$;
