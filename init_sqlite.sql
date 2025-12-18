-- Create the database (if not exists)
-- Note: This is handled by POSTGRES_DB environment variable

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    name TEXT,
    description TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Create metric_values table
CREATE TABLE IF NOT EXISTS metric_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_id INTEGER NOT NULL,
    value REAL NOT NULL,
    source TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (metric_id) REFERENCES metrics (id)
);

-- Create transactions table for financial transactions
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,
    amount REAL NOT NULL,
    description TEXT,
    transaction_date TEXT DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending'
);

-- Create accounts table
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_number TEXT UNIQUE NOT NULL,
    account_type TEXT NOT NULL,
    balance REAL DEFAULT 0.00,
    user_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample revenue metric
INSERT OR IGNORE INTO metrics (id, category, name, description) VALUES (1, 'revenue', 'Revenue', 'Total revenue metrics');

-- Insert sample revenue data
INSERT OR IGNORE INTO metric_values (metric_id, value, source, timestamp) VALUES
(1, 100000.50, 'sales', '2023-01-01 10:00:00'),
(1, 150000.75, 'subscriptions', '2023-01-02 11:00:00'),
(1, 200000.00, 'services', '2023-01-03 12:00:00');

-- Insert sample accounts
INSERT OR IGNORE INTO accounts (account_number, account_type, balance) VALUES
('ACC001', 'checking', 5000.00),
('ACC002', 'savings', 15000.00);

-- Insert sample transactions
INSERT OR IGNORE INTO transactions (account_id, transaction_type, amount, description) VALUES
(1, 'deposit', 1000.00, 'Salary deposit'),
(1, 'withdrawal', 500.00, 'ATM withdrawal'),
(2, 'deposit', 2000.00, 'Interest payment');
