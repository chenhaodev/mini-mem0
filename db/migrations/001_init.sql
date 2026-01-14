-- Initial schema for Homecare Memory
-- Creates memories table with proper indexes for performance

-- Create UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Memories table
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL,
    content TEXT NOT NULL CHECK (length(content) > 0 AND length(content) <= 2000),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE NULL
);

-- Indexes for performance optimization
-- Primary lookup index: patient_id with active memories only
CREATE INDEX IF NOT EXISTS idx_memories_patient_id
    ON memories(patient_id)
    WHERE deleted_at IS NULL;

-- Category filtering
CREATE INDEX IF NOT EXISTS idx_memories_category
    ON memories(category)
    WHERE deleted_at IS NULL;

-- Priority for sorting
CREATE INDEX IF NOT EXISTS idx_memories_priority
    ON memories(priority)
    WHERE deleted_at IS NULL;

-- Composite index for patient + category queries
CREATE INDEX IF NOT EXISTS idx_memories_patient_category
    ON memories(patient_id, category)
    WHERE deleted_at IS NULL;

-- Timestamp for recent observations
CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories(created_at DESC)
    WHERE deleted_at IS NULL;

-- Soft delete filtering
CREATE INDEX IF NOT EXISTS idx_memories_deleted_at
    ON memories(deleted_at)
    WHERE deleted_at IS NOT NULL;

-- JSONB metadata for flexible queries (GIN index)
CREATE INDEX IF NOT EXISTS idx_memories_metadata
    ON memories USING GIN(metadata);

-- Updated_at trigger for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_memories_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE memories IS 'Stores patient memories with categories and priorities for at-home care';
COMMENT ON COLUMN memories.patient_id IS 'Unique identifier for the patient';
COMMENT ON COLUMN memories.category IS 'Memory category: medical_history, allergy, medication, preference, observation, appointment';
COMMENT ON COLUMN memories.priority IS 'Priority level: critical, high, normal';
COMMENT ON COLUMN memories.content IS 'Memory content text (1-2000 characters)';
COMMENT ON COLUMN memories.metadata IS 'Additional structured metadata as JSONB';
COMMENT ON COLUMN memories.deleted_at IS 'Soft delete timestamp (NULL for active memories)';
