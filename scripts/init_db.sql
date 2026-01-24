-- Lameness Detection Database Initialization Script
-- Run this for fresh deployments or to add missing columns to existing deployments

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============== USERS & AUTH ==============

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'rater',
    is_active BOOLEAN DEFAULT TRUE,
    rater_tier VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    CONSTRAINT valid_role CHECK (role IN ('admin', 'researcher', 'rater'))
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============== PROCESSING ==============

CREATE TABLE IF NOT EXISTS processing_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    progress FLOAT DEFAULT 0.0,
    current_pipeline VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    user_id UUID REFERENCES users(id),
    CONSTRAINT valid_job_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_video_id ON processing_jobs(video_id);

-- ============== RATING SYSTEM ==============

CREATE TABLE IF NOT EXISTS gold_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id_1 VARCHAR(100) NOT NULL,
    video_id_2 VARCHAR(100) NOT NULL,
    correct_winner INTEGER NOT NULL,
    correct_degree INTEGER DEFAULT 2,
    difficulty VARCHAR(10) DEFAULT 'medium',
    description TEXT,
    hint TEXT,
    is_tutorial BOOLEAN DEFAULT FALSE,
    tutorial_order INTEGER,
    created_by UUID REFERENCES users(id),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_winner CHECK (correct_winner IN (0, 1, 2)),
    CONSTRAINT valid_difficulty CHECK (difficulty IN ('easy', 'medium', 'hard')),
    CONSTRAINT valid_degree CHECK (correct_degree >= 1 AND correct_degree <= 3)
);

CREATE TABLE IF NOT EXISTS rater_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total_comparisons INTEGER DEFAULT 0,
    gold_task_accuracy FLOAT DEFAULT 0.0,
    agreement_rate FLOAT DEFAULT 0.0,
    weight FLOAT DEFAULT 1.0,
    tier VARCHAR(10) DEFAULT 'bronze',
    last_activity TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_elo_ratings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id VARCHAR(100) UNIQUE NOT NULL,
    elo_rating FLOAT DEFAULT 1500.0,
    elo_uncertainty FLOAT DEFAULT 350.0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    total_comparisons INTEGER DEFAULT 0,
    win_probability FLOAT DEFAULT 0.5,
    normalized_score FLOAT,
    rank_position INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_video_elo_ratings_video_id ON video_elo_ratings(video_id);

CREATE TABLE IF NOT EXISTS pairwise_comparisons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id_1 VARCHAR(100) NOT NULL,
    video_id_2 VARCHAR(100) NOT NULL,
    winner INTEGER NOT NULL,
    degree INTEGER DEFAULT 1,
    confidence VARCHAR(20) DEFAULT 'confident',
    rater_id UUID REFERENCES users(id),
    rater_weight FLOAT DEFAULT 1.0,
    is_gold_task BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_comparison_winner CHECK (winner IN (0, 1, 2)),
    CONSTRAINT valid_degree CHECK (degree >= 0 AND degree <= 3)
);

CREATE INDEX IF NOT EXISTS idx_pairwise_comparisons_video_1 ON pairwise_comparisons(video_id_1);
CREATE INDEX IF NOT EXISTS idx_pairwise_comparisons_video_2 ON pairwise_comparisons(video_id_2);

CREATE TABLE IF NOT EXISTS elo_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id VARCHAR(100) NOT NULL,
    elo_rating FLOAT NOT NULL,
    comparison_count INTEGER NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_elo_history_video_id ON elo_history(video_id);

CREATE TABLE IF NOT EXISTS hierarchy_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    total_videos INTEGER NOT NULL,
    total_comparisons INTEGER NOT NULL,
    steepness FLOAT,
    steepness_std FLOAT,
    inter_rater_reliability FLOAT,
    ranking_data TEXT NOT NULL,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============== SERVICE MONITORING ==============

CREATE TABLE IF NOT EXISTS service_heartbeats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'unknown',
    last_heartbeat TIMESTAMP,
    active_jobs INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    extra_data TEXT
);

CREATE INDEX IF NOT EXISTS idx_service_heartbeats_name ON service_heartbeats(service_name);

-- ============== COW IDENTITY & TRACKING ==============

CREATE TABLE IF NOT EXISTS cow_identities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cow_id VARCHAR(100) UNIQUE NOT NULL,
    tag_number VARCHAR(50),
    total_sightings INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding_version VARCHAR(20) DEFAULT 'dinov3-base',
    notes TEXT,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_cow_identities_cow_id ON cow_identities(cow_id);

CREATE TABLE IF NOT EXISTS track_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_id VARCHAR(100) NOT NULL,
    track_id INTEGER NOT NULL,
    cow_id UUID REFERENCES cow_identities(id),
    reid_confidence FLOAT,
    start_frame INTEGER,
    end_frame INTEGER,
    total_frames INTEGER,
    avg_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_track_history_video_id ON track_history(video_id);
CREATE INDEX IF NOT EXISTS idx_track_history_cow_id ON track_history(cow_id);

CREATE TABLE IF NOT EXISTS lameness_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cow_id UUID NOT NULL REFERENCES cow_identities(id),
    video_id VARCHAR(100) NOT NULL,
    observation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Pipeline scores
    fusion_score FLOAT,
    tleap_score FLOAT,
    tcn_score FLOAT,
    transformer_score FLOAT,
    gnn_score FLOAT,
    graph_transformer_score FLOAT,
    ml_ensemble_score FLOAT,
    
    -- Final prediction
    is_lame BOOLEAN,
    confidence FLOAT,
    severity_level VARCHAR(20),
    
    -- Human validation
    human_validated BOOLEAN DEFAULT FALSE,
    human_label BOOLEAN,
    validator_id UUID REFERENCES users(id),
    validation_date TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lameness_records_cow_id ON lameness_records(cow_id);
CREATE INDEX IF NOT EXISTS idx_lameness_records_video_id ON lameness_records(video_id);

-- ============== MIGRATION: Add missing columns to existing tables ==============
-- These are safe to run multiple times (IF NOT EXISTS)

DO $$ 
BEGIN
    -- Add graph_transformer_score if missing
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name='lameness_records' AND column_name='graph_transformer_score') THEN
        ALTER TABLE lameness_records ADD COLUMN graph_transformer_score FLOAT;
    END IF;
END $$;

-- ============== SEED DATA ==============

-- Default Admin User
-- Email: admin@example.com
-- Password: adminpass123
INSERT INTO users (id, email, username, password_hash, role, is_active, rater_tier, created_at)
VALUES (
    'a0000000-0000-0000-0000-000000000001'::uuid,
    'admin@example.com',
    'admin',
    '$2b$12$RvENahyeLjwOc0WBIounA.A.yRVDG.iDXhFty7nHNnErhg8oXA1j.',
    'admin',
    TRUE,
    'gold',
    CURRENT_TIMESTAMP
) ON CONFLICT (email) DO UPDATE SET created_at = COALESCE(users.created_at, CURRENT_TIMESTAMP);

-- Sample Researcher User
-- Email: researcher@example.com  
-- Password: researcher123
INSERT INTO users (id, email, username, password_hash, role, is_active, rater_tier, created_at)
VALUES (
    'a0000000-0000-0000-0000-000000000002'::uuid,
    'researcher@example.com',
    'researcher',
    '$2b$12$y3RWpoi7LSF1/10ST335g.gaUC72y5qaxH7RCqh3giN51bTLT/hSm',
    'researcher',
    TRUE,
    'gold',
    CURRENT_TIMESTAMP
) ON CONFLICT (email) DO UPDATE SET created_at = COALESCE(users.created_at, CURRENT_TIMESTAMP);

-- Sample Rater User
-- Email: rater@example.com
-- Password: rater123
INSERT INTO users (id, email, username, password_hash, role, is_active, rater_tier, created_at)
VALUES (
    'a0000000-0000-0000-0000-000000000003'::uuid,
    'rater@example.com',
    'rater',
    '$2b$12$8efPMdT57Vw8Zawqya9qG.z5KepzE7PIfIwrG19s7cWPxHJwtk1v.',
    'rater',
    TRUE,
    'bronze',
    CURRENT_TIMESTAMP
) ON CONFLICT (email) DO UPDATE SET created_at = COALESCE(users.created_at, CURRENT_TIMESTAMP);

-- Initialize rater stats for seed users
INSERT INTO rater_stats (id, user_id, total_comparisons, gold_task_accuracy, agreement_rate, weight, tier)
VALUES 
    (uuid_generate_v4(), 'a0000000-0000-0000-0000-000000000001'::uuid, 0, 1.0, 1.0, 1.0, 'gold'),
    (uuid_generate_v4(), 'a0000000-0000-0000-0000-000000000002'::uuid, 0, 1.0, 1.0, 1.0, 'gold'),
    (uuid_generate_v4(), 'a0000000-0000-0000-0000-000000000003'::uuid, 0, 0.0, 0.0, 1.0, 'bronze')
ON CONFLICT (user_id) DO NOTHING;

-- Initialize service heartbeats for monitoring
INSERT INTO service_heartbeats (id, service_name, status, active_jobs, success_count, error_count)
VALUES 
    (uuid_generate_v4(), 'video-ingestion', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'video-preprocessing', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'clip-curation', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'yolo-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'sam3-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'dinov3-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'tleap-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'tcn-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'transformer-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'gnn-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'graph-transformer-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'tracking-service', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'ml-pipeline', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'fusion-service', 'unknown', 0, 0, 0),
    (uuid_generate_v4(), 'shap-service', 'unknown', 0, 0, 0)
ON CONFLICT (service_name) DO NOTHING;

-- Success message
DO $$ BEGIN RAISE NOTICE 'Database initialization complete!'; END $$;

-- ============== SEED DATA SUMMARY ==============
-- 
-- Default Users:
-- +---------------------+------------------+-------------+
-- | Email               | Password         | Role        |
-- +---------------------+------------------+-------------+
-- | admin@example.com   | adminpass123     | admin       |
-- | researcher@example.com | researcher123 | researcher  |
-- | rater@example.com   | rater123         | rater       |
-- +---------------------+------------------+-------------+
--

