-- OWLBAN GROUP AI Database Initialization
-- PostgreSQL initialization script

-- Create database and user (if not exists)
-- Note: This is handled by docker-compose environment variables

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create tables
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    input_data JSONB,
    prediction JSONB,
    confidence DOUBLE PRECISION,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    execution_time DOUBLE PRECISION,
    gpu_used BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS revenue_optimization (
    id SERIAL PRIMARY KEY,
    strategy VARCHAR(255),
    profit DOUBLE PRECISION,
    parameters JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    iterations INTEGER,
    execution_time DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255),
    value DOUBLE PRECISION,
    tags JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS quantum_computations (
    id SERIAL PRIMARY KEY,
    algorithm VARCHAR(255),
    input_parameters JSONB,
    result JSONB,
    execution_time DOUBLE PRECISION,
    quantum_advantage DOUBLE PRECISION,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    qubits_used INTEGER,
    circuit_depth INTEGER
);

CREATE TABLE IF NOT EXISTS gpu_metrics (
    id SERIAL PRIMARY KEY,
    gpu_id INTEGER,
    utilization DOUBLE PRECISION,
    memory_used DOUBLE PRECISION,
    memory_total DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    power_usage DOUBLE PRECISION,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_requests (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time DOUBLE PRECISION,
    user_agent TEXT,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_quantum_computations_algorithm ON quantum_computations(algorithm);
CREATE INDEX IF NOT EXISTS idx_gpu_metrics_gpu_id ON gpu_metrics(gpu_id);
CREATE INDEX IF NOT EXISTS idx_gpu_metrics_timestamp ON gpu_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_requests_timestamp ON api_requests(timestamp);

-- Create views for analytics
CREATE OR REPLACE VIEW daily_metrics AS
SELECT
    DATE(timestamp) as date,
    metric_name,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    COUNT(*) as count
FROM system_metrics
GROUP BY DATE(timestamp), metric_name
ORDER BY date DESC, metric_name;

CREATE OR REPLACE VIEW api_performance AS
SELECT
    DATE(timestamp) as date,
    endpoint,
    AVG(response_time) as avg_response_time,
    COUNT(*) as request_count,
    AVG(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_rate
FROM api_requests
GROUP BY DATE(timestamp), endpoint
ORDER BY date DESC, endpoint;

-- Insert sample data
INSERT INTO system_metrics (metric_name, value, tags, source) VALUES
('cpu_usage', 45.5, '{"server": "api-server"}', 'monitoring'),
('memory_usage', 67.8, '{"server": "api-server"}', 'monitoring'),
('gpu_utilization', 89.2, '{"gpu": 0}', 'dcgm'),
('api_response_time', 0.023, '{"endpoint": "/inference"}', 'api_server')
ON CONFLICT DO NOTHING;

-- Create user and grant permissions (if needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO owlban;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO owlban;
