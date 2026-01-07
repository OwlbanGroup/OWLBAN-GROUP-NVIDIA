-- ===========================================
-- JPMORGAN FINANCIAL DATA TABLES
-- Component 3: Database Schema (PostgreSQL) - JPMorgan-Specific Tables
-- ===========================================

-- Enable necessary extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===========================================
-- JPMORGAN ACCOUNTS TABLE
-- ===========================================

-- JPMorgan Accounts (synchronized from API)
CREATE TABLE IF NOT EXISTS jpmorgan_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    jpmorgan_id VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    currency_code CHAR(3) DEFAULT 'USD',
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'closed')),
    account_metadata JSONB,
    last_sync TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- JPMORGAN BALANCES TABLE
-- ===========================================

-- JPMorgan Balances (daily snapshots)
CREATE TABLE IF NOT EXISTS jpmorgan_balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES jpmorgan_accounts(id) ON DELETE CASCADE,
    available_balance DECIMAL(20,4),
    ledger_balance DECIMAL(20,4) NOT NULL,
    currency_code CHAR(3) DEFAULT 'USD',
    balance_date DATE NOT NULL,
    balance_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, balance_date)
);

-- ===========================================
-- JPMORGAN TRANSACTIONS TABLE
-- ===========================================

-- JPMorgan Transactions
CREATE TABLE IF NOT EXISTS jpmorgan_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    jpmorgan_id VARCHAR(255) NOT NULL UNIQUE,
    account_id UUID NOT NULL REFERENCES jpmorgan_accounts(id) ON DELETE CASCADE,
    amount DECIMAL(20,4) NOT NULL,
    currency_code CHAR(3) DEFAULT 'USD',
    transaction_type VARCHAR(100) NOT NULL,
    description TEXT,
    transaction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    posting_date TIMESTAMP WITH TIME ZONE,
    reference_number VARCHAR(255),
    check_number VARCHAR(50),
    payee_name VARCHAR(255),
    category VARCHAR(100),
    subcategory VARCHAR(100),
    status VARCHAR(50) DEFAULT 'posted' CHECK (status IN ('pending', 'posted', 'failed', 'cancelled')),
    transaction_metadata JSONB,
    last_sync TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================
-- SYNC LOGS TABLE
-- ===========================================

-- Sync Logs (track all API synchronization operations)
CREATE TABLE IF NOT EXISTS sync_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sync_type VARCHAR(50) NOT NULL CHECK (sync_type IN ('accounts', 'balances', 'transactions', 'full')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('started', 'running', 'completed', 'failed')),
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    error_message TEXT,
    sync_metadata JSONB,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER
);

-- ===========================================
-- INDEXES FOR PERFORMANCE
-- ===========================================

-- JPMorgan Accounts indexes
CREATE INDEX IF NOT EXISTS idx_jpmorgan_accounts_jpmorgan_id ON jpmorgan_accounts(jpmorgan_id);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_accounts_type ON jpmorgan_accounts(type);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_accounts_status ON jpmorgan_accounts(status);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_accounts_last_sync ON jpmorgan_accounts(last_sync);

-- JPMorgan Balances indexes
CREATE INDEX IF NOT EXISTS idx_jpmorgan_balances_account_id ON jpmorgan_balances(account_id);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_balances_date ON jpmorgan_balances(balance_date);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_balances_account_date ON jpmorgan_balances(account_id, balance_date);

-- JPMorgan Transactions indexes
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_jpmorgan_id ON jpmorgan_transactions(jpmorgan_id);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_account_id ON jpmorgan_transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_date ON jpmorgan_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_type ON jpmorgan_transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_status ON jpmorgan_transactions(status);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_amount ON jpmorgan_transactions(amount);
CREATE INDEX IF NOT EXISTS idx_jpmorgan_transactions_last_sync ON jpmorgan_transactions(last_sync);

-- Sync Logs indexes
CREATE INDEX IF NOT EXISTS idx_sync_logs_type ON sync_logs(sync_type);
CREATE INDEX IF NOT EXISTS idx_sync_logs_status ON sync_logs(status);
CREATE INDEX IF NOT EXISTS idx_sync_logs_started_at ON sync_logs(started_at);
CREATE INDEX IF NOT EXISTS idx_sync_logs_completed_at ON sync_logs(completed_at);

-- ===========================================
-- VIEWS FOR COMMON QUERIES
-- ===========================================

-- Current account balances view
CREATE OR REPLACE VIEW jpmorgan_current_balances AS
SELECT
    a.id,
    a.jpmorgan_id,
    a.name,
    a.type,
    a.currency_code,
    COALESCE(b.available_balance, 0) as available_balance,
    COALESCE(b.ledger_balance, 0) as ledger_balance,
    b.balance_date as last_balance_date,
    a.last_sync
FROM jpmorgan_accounts a
LEFT JOIN jpmorgan_balances b ON a.id = b.account_id
    AND b.balance_date = (
        SELECT MAX(balance_date)
        FROM jpmorgan_balances
        WHERE account_id = a.id
    )
WHERE a.status = 'active';

-- Recent transactions view (last 30 days)
CREATE OR REPLACE VIEW jpmorgan_recent_transactions AS
SELECT
    t.id,
    t.jpmorgan_id,
    t.account_id,
    a.name as account_name,
    a.type as account_type,
    t.amount,
    t.currency_code,
    t.transaction_type,
    t.description,
    t.transaction_date,
    t.status,
    t.category,
    t.subcategory
FROM jpmorgan_transactions t
JOIN jpmorgan_accounts a ON t.account_id = a.id
WHERE t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
    AND t.status = 'posted'
ORDER BY t.transaction_date DESC;

-- Daily transaction summary
CREATE OR REPLACE VIEW jpmorgan_daily_transaction_summary AS
SELECT
    DATE(t.transaction_date) as transaction_date,
    a.type as account_type,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN t.amount > 0 THEN t.amount ELSE 0 END) as total_credits,
    SUM(CASE WHEN t.amount < 0 THEN ABS(t.amount) ELSE 0 END) as total_debits,
    AVG(ABS(t.amount)) as avg_transaction_amount
FROM jpmorgan_transactions t
JOIN jpmorgan_accounts a ON t.account_id = a.id
WHERE t.status = 'posted'
    AND t.transaction_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(t.transaction_date), a.type
ORDER BY transaction_date DESC;

-- Sync status summary
CREATE OR REPLACE VIEW sync_status_summary AS
SELECT
    sync_type,
    status,
    COUNT(*) as count,
    MAX(started_at) as last_sync_attempt,
    AVG(duration_seconds) as avg_duration_seconds,
    SUM(records_processed) as total_records_processed,
    SUM(records_failed) as total_records_failed
FROM sync_logs
WHERE started_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY sync_type, status
ORDER BY sync_type, status;

-- ===========================================
-- TRIGGERS FOR AUDIT LOGGING
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Update triggers for JPMorgan tables
DROP TRIGGER IF EXISTS update_jpmorgan_accounts_updated_at ON jpmorgan_accounts;
CREATE TRIGGER update_jpmorgan_accounts_updated_at
    BEFORE UPDATE ON jpmorgan_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_jpmorgan_transactions_updated_at ON jpmorgan_transactions;
CREATE TRIGGER update_jpmorgan_transactions_updated_at
    BEFORE UPDATE ON jpmorgan_transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- SAMPLE DATA INSERTION
-- ===========================================

-- Insert sample JPMorgan accounts (only if table is empty)
INSERT INTO jpmorgan_accounts (jpmorgan_id, name, type, currency_code, status, account_metadata)
SELECT 'JPM-ACC-001', 'JPMorgan Checking Account', 'checking', 'USD', 'active', '{"branch": "New York", "routing_number": "021000021"}'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-001');

INSERT INTO jpmorgan_accounts (jpmorgan_id, name, type, currency_code, status, account_metadata)
SELECT 'JPM-ACC-002', 'JPMorgan Savings Account', 'savings', 'USD', 'active', '{"interest_rate": "0.005", "minimum_balance": "1000"}'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-002');

INSERT INTO jpmorgan_accounts (jpmorgan_id, name, type, currency_code, status, account_metadata)
SELECT 'JPM-ACC-003', 'JPMorgan Investment Account', 'investment', 'USD', 'active', '{"portfolio_type": "diversified", "risk_level": "moderate"}'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-003');

-- Insert sample balances (only if not exists for today)
INSERT INTO jpmorgan_balances (account_id, available_balance, ledger_balance, currency_code, balance_date)
SELECT
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-001'),
    25000.00, 25000.00, 'USD', CURRENT_DATE
WHERE NOT EXISTS (
    SELECT 1 FROM jpmorgan_balances
    WHERE account_id = (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-001')
    AND balance_date = CURRENT_DATE
);

INSERT INTO jpmorgan_balances (account_id, available_balance, ledger_balance, currency_code, balance_date)
SELECT
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-002'),
    50000.00, 50000.00, 'USD', CURRENT_DATE
WHERE NOT EXISTS (
    SELECT 1 FROM jpmorgan_balances
    WHERE account_id = (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-002')
    AND balance_date = CURRENT_DATE
);

INSERT INTO jpmorgan_balances (account_id, available_balance, ledger_balance, currency_code, balance_date)
SELECT
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-003'),
    150000.00, 150000.00, 'USD', CURRENT_DATE
WHERE NOT EXISTS (
    SELECT 1 FROM jpmorgan_balances
    WHERE account_id = (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-003')
    AND balance_date = CURRENT_DATE
);

-- Insert sample transactions (only if not exists)
INSERT INTO jpmorgan_transactions (
    jpmorgan_id, account_id, amount, currency_code, transaction_type,
    description, transaction_date, status, category
)
SELECT 'JPM-TX-001',
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-001'),
    -150.00, 'USD', 'debit', 'Grocery Store Purchase',
    CURRENT_DATE - INTERVAL '1 day', 'posted', 'shopping'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_transactions WHERE jpmorgan_id = 'JPM-TX-001');

INSERT INTO jpmorgan_transactions (
    jpmorgan_id, account_id, amount, currency_code, transaction_type,
    description, transaction_date, status, category
)
SELECT 'JPM-TX-002',
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-001'),
    3000.00, 'USD', 'credit', 'Salary Deposit',
    CURRENT_DATE - INTERVAL '2 days', 'posted', 'income'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_transactions WHERE jpmorgan_id = 'JPM-TX-002');

INSERT INTO jpmorgan_transactions (
    jpmorgan_id, account_id, amount, currency_code, transaction_type,
    description, transaction_date, status, category
)
SELECT 'JPM-TX-003',
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-002'),
    500.00, 'USD', 'credit', 'Interest Payment',
    CURRENT_DATE - INTERVAL '1 day', 'posted', 'interest'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_transactions WHERE jpmorgan_id = 'JPM-TX-003');

INSERT INTO jpmorgan_transactions (
    jpmorgan_id, account_id, amount, currency_code, transaction_type,
    description, transaction_date, status, category
)
SELECT 'JPM-TX-004',
    (SELECT id FROM jpmorgan_accounts WHERE jpmorgan_id = 'JPM-ACC-003'),
    2500.00, 'USD', 'credit', 'Dividend Payment',
    CURRENT_DATE - INTERVAL '3 days', 'posted', 'dividend'
WHERE NOT EXISTS (SELECT 1 FROM jpmorgan_transactions WHERE jpmorgan_id = 'JPM-TX-004');

-- Insert sample sync log
INSERT INTO sync_logs (sync_type, status, records_processed, started_at, completed_at, duration_seconds)
SELECT 'full', 'completed', 7,
    CURRENT_TIMESTAMP - INTERVAL '1 hour',
    CURRENT_TIMESTAMP - INTERVAL '55 minutes', 300
WHERE NOT EXISTS (SELECT 1 FROM sync_logs WHERE sync_type = 'full' AND status = 'completed');

-- ===========================================
-- PERMISSIONS AND SECURITY
-- ===========================================

-- Create roles (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jpmorgan_readonly') THEN
        CREATE ROLE jpmorgan_readonly;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jpmorgan_analyst') THEN
        CREATE ROLE jpmorgan_analyst;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'jpmorgan_admin') THEN
        CREATE ROLE jpmorgan_admin;
    END IF;
END $$;

-- Grant permissions for JPMorgan tables
GRANT SELECT ON jpmorgan_accounts TO jpmorgan_readonly;
GRANT SELECT ON jpmorgan_balances TO jpmorgan_readonly;
GRANT SELECT ON jpmorgan_transactions TO jpmorgan_readonly;
GRANT SELECT ON sync_logs TO jpmorgan_readonly;

GRANT SELECT, INSERT, UPDATE ON jpmorgan_accounts TO jpmorgan_analyst;
GRANT SELECT, INSERT, UPDATE ON jpmorgan_balances TO jpmorgan_analyst;
GRANT SELECT, INSERT, UPDATE ON jpmorgan_transactions TO jpmorgan_analyst;
GRANT SELECT, INSERT ON sync_logs TO jpmorgan_analyst;

GRANT ALL PRIVILEGES ON jpmorgan_accounts TO jpmorgan_admin;
GRANT ALL PRIVILEGES ON jpmorgan_balances TO jpmorgan_admin;
GRANT ALL PRIVILEGES ON jpmorgan_transactions TO jpmorgan_admin;
GRANT ALL PRIVILEGES ON sync_logs TO jpmorgan_admin;

-- Grant permissions on sequences
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO jpmorgan_analyst;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO jpmorgan_admin;

-- ===========================================
-- TABLE COMMENTS
-- ===========================================

COMMENT ON TABLE jpmorgan_accounts IS 'JPMorgan bank and investment accounts synchronized from API';
COMMENT ON TABLE jpmorgan_balances IS 'Daily balance snapshots for JPMorgan accounts';
COMMENT ON TABLE jpmorgan_transactions IS 'Transaction history synchronized from JPMorgan APIs';
COMMENT ON TABLE sync_logs IS 'Audit log of all JPMorgan API synchronization operations';

-- ===========================================
-- USAGE INSTRUCTIONS
-- ===========================================

/*
To apply this schema to your existing database:

1. Connect to your PostgreSQL database
2. Run: \i jpmorgan_database_schema.sql

Or from command line:
psql -d your_database -f jpmorgan_database_schema.sql

This schema is designed to work alongside the existing database_schema.sql
and adds JPMorgan-specific tables for API synchronization.
*/
