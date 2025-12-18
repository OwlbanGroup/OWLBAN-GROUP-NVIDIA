-- Create the database (if not exists)
-- Note: This is handled by POSTGRES_DB environment variable

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    category TEXT NOT NULL,
    name TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create metric_values table
CREATE TABLE IF NOT EXISTS metric_values (
    id SERIAL PRIMARY KEY,
    metric_id INTEGER NOT NULL,
    value REAL NOT NULL,
    source TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (metric_id) REFERENCES metrics (id)
);

-- Create transactions table for financial transactions
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    description TEXT,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending'
);

-- Create accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id SERIAL PRIMARY KEY,
    account_number TEXT UNIQUE NOT NULL,
    account_type TEXT NOT NULL,
    balance DECIMAL(15,2) DEFAULT 0.00,
    user_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample revenue metric
INSERT INTO metrics (id, category, name, description) VALUES (1, 'revenue', 'Revenue', 'Total revenue metrics')
ON CONFLICT (id) DO NOTHING;

-- Insert sample revenue data
INSERT INTO metric_values (metric_id, value, source, timestamp) VALUES
(1, 100000.50, 'sales', '2023-01-01 10:00:00'),
(1, 150000.75, 'subscriptions', '2023-01-02 11:00:00'),
(1, 200000.00, 'services', '2023-01-03 12:00:00')
ON CONFLICT DO NOTHING;

-- Insert sample accounts
INSERT INTO accounts (account_number, account_type, balance) VALUES
('ACC001', 'checking', 5000.00),
('ACC002', 'savings', 15000.00)
ON CONFLICT (account_number) DO NOTHING;

-- Insert sample transactions
INSERT INTO transactions (account_id, transaction_type, amount, description) VALUES
(1, 'deposit', 1000.00, 'Salary deposit'),
(1, 'withdrawal', 500.00, 'ATM withdrawal'),
(2, 'deposit', 2000.00, 'Interest payment')
ON CONFLICT DO NOTHING;
