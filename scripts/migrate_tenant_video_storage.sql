-- Migration: Tenant-prefixed video storage + new roles + clip_videos
-- Run once on existing DB. Safe to re-run (IF NOT EXISTS / DROP IF EXISTS).

-- 1) Extend user roles: viewer, tenant_admin
ALTER TABLE users DROP CONSTRAINT IF EXISTS valid_role;
ALTER TABLE users ADD CONSTRAINT valid_role CHECK (role IN ('admin', 'tenant_admin', 'researcher', 'viewer', 'rater'));

-- 2) Create clip_videos table
CREATE TABLE IF NOT EXISTS clip_videos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL REFERENCES tenants(id),
    raw_video_id VARCHAR(36) NOT NULL REFERENCES videos(id),
    clip_slug VARCHAR(20) NOT NULL,
    s3_key VARCHAR(512) NOT NULL UNIQUE,
    start_sec FLOAT,
    end_sec FLOAT,
    quality_label VARCHAR(50),
    label VARCHAR(10),
    prob_bad FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_clip_videos_raw_slug UNIQUE (raw_video_id, clip_slug)
);

CREATE INDEX IF NOT EXISTS idx_clip_videos_tenant ON clip_videos(tenant_id);
CREATE INDEX IF NOT EXISTS idx_clip_videos_raw ON clip_videos(raw_video_id);
