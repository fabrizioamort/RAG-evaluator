-- Initial database setup for RAG Evaluation Platform
-- This script runs on first PostgreSQL container start

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The database and user are created via environment variables
-- This file is for any additional initialization

-- Grant all privileges to the application user
GRANT ALL PRIVILEGES ON DATABASE rageval TO rageval;

-- Set default search path
ALTER DATABASE rageval SET search_path TO public;
