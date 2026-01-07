-- JPMorgan Financial APIs - Core Database Schema
-- Production-ready PostgreSQL schema for financial data management

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===========================================
-- CORE ENTITIES TABLES
-- ===========================================

-- Entities (Companies, Organizations, Individuals)
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL CHECK (entity_type IN ('company', 'individual', 'organization', 'trust')),
    name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(255),
    tax_id VARCHAR(100) UNIQUE,
    registration_number VARCHAR(100),
    incorporation_date DATE,
    business_address JSONB,
    mailing_address JSONB,
    contact_info JSONB,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended', 'dissolved')),
    risk_rating VARCHAR(20) DEFAULT 'low' CHECK (risk_rating IN ('low', 'medium', 'high', 'critical')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID,
    updated_by UUID
);

-- Accounts (Bank accounts, investment accounts, etc.)
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    account_type VARCHAR(100) NOT NULL,
    account_number VARCHAR(100) UNIQUE NOT NULL,
    account_name VARCHAR(255),
    bank_name VARCHAR(255),
    bank_routing_number VARCHAR(50),
    currency_code CHAR(3) DEFAULT 'USD',
    account_status VARCHAR(50) DEFAULT 'active' CHECK (account_status IN ('active', 'closed', 'frozen', 'pending')),
    opening_date DATE,
    closing_date DATE,
    account_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- TRANSACTION MANAGEMENT
-- ===========================================

-- Transactions (All financial transactions)
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    transaction_type VARCHAR(100) NOT NULL,
    amount DECIMAL(20,4) NOT NULL,
    currency_code CHAR(3) DEFAULT 'USD',
    transaction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    posting_date TIMESTAMP WITH TIME ZONE,
    description TEXT,
    reference_number VARCHAR(255),
    check_number VARCHAR(50),
    payee_name VARCHAR(255),
    payee_account VARCHAR(100),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    tags TEXT[],
    transaction_metadata JSONB,
    status VARCHAR(50) DEFAULT 'posted' CHECK (status IN ('pending', 'posted', 'failed', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- BALANCE MANAGEMENT
-- ===========================================

-- Account Balances (Daily balance snapshots)
CREATE TABLE balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    balance_date DATE NOT NULL,
    opening_balance DECIMAL(20,4),
    closing_balance DECIMAL(20,4) NOT NULL,
    available_balance DECIMAL(20,4),
    pending_balance DECIMAL(20,4),
    currency_code CHAR(3) DEFAULT 'USD',
    balance_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, balance_date)
);

-- ===========================================
-- PAYMENT MANAGEMENT
-- ===========================================

-- Scheduled Payments
CREATE TABLE scheduled_payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    payment_type VARCHAR(100) NOT NULL,
    amount DECIMAL(20,4) NOT NULL,
    currency_code CHAR(3) DEFAULT 'USD',
    payee_name VARCHAR(255) NOT NULL,
    payee_account VARCHAR(100),
    payment_date DATE NOT NULL,
    frequency VARCHAR(50) CHECK (frequency IN ('one-time', 'weekly', 'bi-weekly', 'monthly', 'quarterly', 'annually')),
    next_payment_date DATE,
    end_date DATE,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'cancelled', 'failed')),
    payment_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- ALERTS & MONITORING
-- ===========================================

-- Alerts (System and user-defined alerts)
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    entity_id UUID REFERENCES entities(id),
    account_id UUID REFERENCES accounts(id),
    transaction_id UUID REFERENCES transactions(id),
    threshold_value DECIMAL(20,4),
    actual_value DECIMAL(20,4),
    alert_metadata JSONB,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved', 'dismissed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID
);

-- ===========================================
-- AUDIT & COMPLIANCE
-- ===========================================

-- Audit Log (All system activities)
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_by UUID,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- ===========================================
-- INDEXES FOR PERFORMANCE
-- ===========================================

-- Entity indexes
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_status ON entities(status);
CREATE INDEX idx_entities_tax_id ON entities(tax_id);

-- Account indexes
CREATE INDEX idx_accounts_entity_id ON accounts(entity_id);
CREATE INDEX idx_accounts_type ON accounts(account_type);
CREATE INDEX idx_accounts_status ON accounts(account_status);

-- Transaction indexes
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_transactions_type ON transactions(transaction_type);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_transactions_status ON transactions(status);

-- Balance indexes
CREATE INDEX idx_balances_account_id ON balances(account_id);
CREATE INDEX idx_balances_date ON balances(balance_date);

-- Alert indexes
CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_entity_id ON alerts(entity_id);
CREATE INDEX idx_alerts_account_id ON alerts(account_id);

-- ===========================================
-- TRIGGERS FOR AUDIT LOGGING
-- ===========================================

-- Function to create audit log entries
CREATE OR REPLACE FUNCTION audit_trigger_function() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_values, changed_at)
        VALUES (TG_TABLE_NAME, OLD.id, TG_OP, row_to_json(OLD), CURRENT_TIMESTAMP);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, operation, old_values, new_values, changed_at)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(OLD), row_to_json(NEW), CURRENT_TIMESTAMP);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, operation, new_values, changed_at)
        VALUES (TG_TABLE_NAME, NEW.id, TG_OP, row_to_json(NEW), CURRENT_TIMESTAMP);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for key tables
CREATE TRIGGER audit_entities AFTER INSERT OR UPDATE OR DELETE ON entities
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_accounts AFTER INSERT OR UPDATE OR DELETE ON accounts
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_transactions AFTER INSERT OR UPDATE OR DELETE ON transactions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_balances AFTER INSERT OR UPDATE OR DELETE ON balances
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- ===========================================
-- VIEWS FOR COMMON QUERIES
-- ===========================================

-- Account summary view
CREATE VIEW account_summary AS
SELECT
    a.id,
    a.account_number,
    a.account_name,
    a.account_type,
    a.currency_code,
    a.account_status,
    e.name as entity_name,
    e.entity_type,
    COALESCE(b.closing_balance, 0) as current_balance,
    COALESCE(b.available_balance, 0) as available_balance,
    b.balance_date as last_balance_date
FROM accounts a
JOIN entities e ON a.entity_id = e.id
LEFT JOIN balances b ON a.id = b.account_id
    AND b.balance_date = (
        SELECT MAX(balance_date)
        FROM balances
        WHERE account_id = a.id
    );

-- Monthly transaction summary
CREATE VIEW monthly_transaction_summary AS
SELECT
    DATE_TRUNC('month', transaction_date) as month,
    account_id,
    transaction_type,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM transactions
WHERE status = 'posted'
GROUP BY DATE_TRUNC('month', transaction_date), account_id, transaction_type
ORDER BY month DESC, account_id;

-- ===========================================
-- SAMPLE DATA INSERTION
-- ===========================================

-- Insert sample entity
INSERT INTO entities (entity_type, name, legal_name, tax_id, status) VALUES
('company', 'JPMorgan Sample Corp', 'JPMorgan Sample Corporation', '12-3456789', 'active');

-- Insert sample account
INSERT INTO accounts (entity_id, account_type, account_number, account_name, bank_name, currency_code) VALUES
((SELECT id FROM entities WHERE name = 'JPMorgan Sample Corp'), 'checking', '1234567890', 'Primary Checking', 'JPMorgan Chase', 'USD');

-- ===========================================
-- PERMISSIONS AND SECURITY
-- ===========================================

-- Create roles
CREATE ROLE jpmorgan_readonly;
CREATE ROLE jpmorgan_analyst;
CREATE ROLE jpmorgan_admin;

-- Grant permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO jpmorgan_readonly;
GRANT SELECT, INSERT, UPDATE ON entities, accounts, transactions, balances TO jpmorgan_analyst;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO jpmorgan_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO jpmorgan_admin;

-- Row Level Security (RLS) for multi-tenant access
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE balances ENABLE ROW LEVEL SECURITY;

-- Create RLS policies (example - customize based on your auth system)
-- CREATE POLICY entity_access ON entities FOR ALL USING (entity_id IN (SELECT entity_id FROM user_entities WHERE user_id = current_user_id));

-- ===========================================
-- PARTITIONING (FOR LARGE DATASETS)
-- ===========================================

-- Partition transactions table by month (for performance with large datasets)
-- CREATE TABLE transactions_y2024m01 PARTITION OF transactions FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- Add more partitions as needed

COMMENT ON DATABASE CURRENT_DATABASE IS 'JPMorgan Financial APIs - Production Database';
COMMENT ON SCHEMA public IS 'Core financial data schema for JPMorgan-style financial management system';
