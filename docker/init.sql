-- 初始化 pgvector 扩展（首次启动容器自动执行）
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. 商品主数据表（严格区分主商品与配件，支撑强元数据过滤与防串品）
CREATE TABLE IF NOT EXISTS products (
    sku_id          TEXT PRIMARY KEY,
    asin            TEXT UNIQUE,
    title           TEXT NOT NULL,
    category        TEXT NOT NULL,
    sub_category    TEXT,
    brand           TEXT,
    is_accessory    BOOLEAN NOT NULL DEFAULT FALSE,
    price_usd       NUMERIC(10, 2),
    rating_avg      NUMERIC(3, 2),
    review_count    INTEGER DEFAULT 0,
    specs_json      JSONB NOT NULL DEFAULT '{}'::jsonb,
    target_pets     TEXT[] DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. 商品评论与知识片段表（支持增量 upsert、逻辑删除、内容哈希去重与 HNSW 向量索引）
CREATE TABLE IF NOT EXISTS review_chunks (
    chunk_id            TEXT PRIMARY KEY,
    sku_id              TEXT NOT NULL REFERENCES products(sku_id) ON DELETE CASCADE,
    review_id           TEXT NOT NULL,
    source_type         TEXT NOT NULL DEFAULT 'review', -- review | qa | manual
    content             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    rating              SMALLINT,
    verified_purchase   BOOLEAN NOT NULL DEFAULT FALSE,
    helpful_votes       INTEGER NOT NULL DEFAULT 0,
    review_date         DATE,
    quality_score       REAL NOT NULL DEFAULT 1.0,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    embedding           vector(1024), -- 默认兼容主流 1024 维（如 BGE-M3 / Qwen text-embedding-v3），可按模型调整
    tsv                 TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (sku_id, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_review_chunks_sku_active ON review_chunks (sku_id, is_active);
CREATE INDEX IF NOT EXISTS idx_review_chunks_tsv ON review_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_review_chunks_embedding_hnsw
    ON review_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 3. 线上问答审计与 Badcase 反馈表（支撑产业自升级闭环）
CREATE TABLE IF NOT EXISTS qa_feedback_logs (
    id                  BIGSERIAL PRIMARY KEY,
    trace_id            TEXT NOT NULL,
    pipeline_version    TEXT NOT NULL, -- v1_vector | v2_agentic_hybrid
    sku_id              TEXT,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    abstained           BOOLEAN NOT NULL DEFAULT FALSE,
    retrieved_count     INTEGER NOT NULL DEFAULT 0,
    latency_ms          INTEGER NOT NULL DEFAULT 0,
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    user_feedback       SMALLINT, -- 1=好评, -1=差评, NULL=未反馈
    badcase_tag         TEXT,     -- wrong_sku | hallucination | missed_recall | over_abstain
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
