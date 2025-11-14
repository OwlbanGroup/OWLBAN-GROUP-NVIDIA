-- PostgreSQL initialization script for JPMorgan Financial APIs
-- This script sets up the production database schema

-- Create database if it doesn't exist
-- Note: This is handled by docker-compose environment variables

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom types
CREATE TYPE user_role AS ENUM ('admin', 'user', 'analyst');
CREATE TYPE asset_status AS ENUM ('active', 'inactive', 'maintenance', 'retired');
CREATE TYPE asset_type AS ENUM ('server', 'workstation', 'network_device', 'storage', 'other');

-- Create users table (for production use)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role user_role DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    token_expires_at TIMESTAMP WITH TIME ZONE
);

-- Create businesses table
CREATE TABLE IF NOT EXISTS businesses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    industry VARCHAR(100),
    location VARCHAR(255),
    contact_email VARCHAR(255),
    website VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Create assets table
CREATE TABLE IF NOT EXISTS assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    asset_type asset_type NOT NULL,
    value DECIMAL(15,2),
    location VARCHAR(255),
    status asset_status DEFAULT 'active',
    purchase_date DATE,
    acquisition_date DATE,
    business_id UUID REFERENCES businesses(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Create telemetry_events table
CREATE TABLE IF NOT EXISTS telemetry_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_name VARCHAR(500) NOT NULL,
    event_version VARCHAR(10) DEFAULT '4.0',
    event_time TIMESTAMP WITH TIME ZONE NOT NULL,
    operation VARCHAR(500),
    pfn VARCHAR(500),
    os VARCHAR(100),
    device_model VARCHAR(255),
    user_id VARCHAR(255),
    raw_data JSONB,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    anomaly_score DECIMAL(5,4),
    is_anomaly BOOLEAN DEFAULT false,
    business_id UUID REFERENCES businesses(id)
);

-- Create ml_models table
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    parameters JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES users(id)
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_telemetry_events_time ON telemetry_events(event_time);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_anomaly ON telemetry_events(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_telemetry_events_business ON telemetry_events(business_id);
CREATE INDEX IF NOT EXISTS idx_assets_business ON assets(business_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_businesses_updated_at BEFORE UPDATE ON businesses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_assets_updated_at BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    action_type TEXT;
    resource_id UUID;
BEGIN
    -- Determine action type
    IF TG_OP = 'INSERT' THEN
        action_type := 'CREATE';
        resource_id := NEW.id;
    ELSIF TG_OP = 'UPDATE' THEN
        action_type := 'UPDATE';
        resource_id := NEW.id;
    ELSIF TG_OP = 'DELETE' THEN
        action_type := 'DELETE';
        resource_id := OLD.id;
    END IF;

    -- Insert audit record
    INSERT INTO audit_log (
        user_id,
        action,
        resource_type,
        resource_id,
        details,
        created_at
    ) VALUES (
        NULL, -- We don't have user context in triggers
        action_type,
        TG_TABLE_NAME,
        resource_id,
        row_to_json(CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END),
        CURRENT_TIMESTAMP
    );

    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers (optional - uncomment if audit logging is needed)
-- CREATE TRIGGER audit_businesses AFTER INSERT OR UPDATE OR DELETE ON businesses
--     FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
--
-- CREATE TRIGGER audit_assets AFTER INSERT OR UPDATE OR DELETE ON assets
--     FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Insert default admin user (change password in production!)
-- Password hash for 'admin123' - CHANGE THIS IN PRODUCTION!
INSERT INTO users (username, email, password_hash, role)
VALUES ('admin', 'admin@jpmorgan.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6fM9q7F6', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Create views for analytics
CREATE OR REPLACE VIEW telemetry_summary AS
SELECT
    DATE_TRUNC('day', event_time) AS date,
    COUNT(*) AS total_events,
    COUNT(*) FILTER (WHERE is_anomaly = true) AS anomaly_events,
    AVG(anomaly_score) FILTER (WHERE anomaly_score IS NOT NULL) AS avg_anomaly_score,
    COUNT(DISTINCT user_id) AS unique_users
FROM telemetry_events
GROUP BY DATE_TRUNC('day', event_time)
ORDER BY date DESC;

CREATE OR REPLACE VIEW asset_summary AS
SELECT
    b.name AS business_name,
    COUNT(a.*) AS total_assets,
    SUM(a.value) AS total_value,
    COUNT(*) FILTER (WHERE a.status = 'active') AS active_assets,
    COUNT(*) FILTER (WHERE a.status = 'maintenance') AS maintenance_assets
FROM businesses b
LEFT JOIN assets a ON b.id = a.business_id
GROUP BY b.id, b.name
ORDER BY total_value DESC NULLS LAST;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO jpmorgan_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO jpmorgan_app;

-- Create backup function
CREATE OR REPLACE FUNCTION create_backup()
RETURNS TEXT AS $$
DECLARE
    backup_filename TEXT;
BEGIN
    backup_filename := 'backup_' || to_char(CURRENT_TIMESTAMP, 'YYYYMMDD_HH24MI') || '.sql';
    EXECUTE format('COPY (SELECT * FROM telemetry_events ORDER BY event_time DESC LIMIT 10000) TO ''/backups/%s'' WITH CSV HEADER', backup_filename);
    RETURN 'Backup created: ' || backup_filename;
END;
$$ LANGUAGE plpgsql;
