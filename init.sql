-- Initialize database with proper user and permissions
CREATE USER IF NOT EXISTS jpmorgan WITH PASSWORD 'secure_password_123';
GRANT ALL PRIVILEGES ON DATABASE jpmorgan_api TO jpmorgan;
ALTER USER jpmorgan CREATEDB;

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
