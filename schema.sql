-- Fog of War: Schema for news corpus and LLM evaluation

-- Raw articles collected from news sources
CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    title           TEXT NOT NULL,
    title_normalized TEXT NOT NULL,  -- lowercase, trimmed for dedup
    url             TEXT,
    google_url      TEXT,
    source_domain   TEXT NOT NULL,   -- e.g. 'reuters.com'
    source_name     TEXT NOT NULL,   -- e.g. 'Reuters'
    published_at    TIMESTAMPTZ,     -- article publication timestamp
    full_text       TEXT,            -- scraped article body
    text_length     INTEGER,         -- character count of full_text
    word_count      INTEGER,         -- word count of full_text
    scrape_status   TEXT DEFAULT 'pending',  -- pending, success, failed, paywall
    scrape_error    TEXT,
    scraped_at      TIMESTAMPTZ,
    query_source    TEXT,            -- which search query found this
    raw_metadata    JSONB,           -- original Google News / GDELT metadata
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_title UNIQUE (title_normalized, source_domain)
);

-- Indexes for fast temporal queries
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_domain);
CREATE INDEX IF NOT EXISTS idx_articles_source_time ON articles(source_domain, published_at);
CREATE INDEX IF NOT EXISTS idx_articles_scrape_status ON articles(scrape_status);
CREATE INDEX IF NOT EXISTS idx_articles_text_search ON articles USING gin(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(full_text, '')));

-- The 11 temporal nodes from the paper
CREATE TABLE IF NOT EXISTS temporal_nodes (
    id              SERIAL PRIMARY KEY,
    node_index      INTEGER NOT NULL UNIQUE,  -- T0, T1, ... T10
    label           TEXT NOT NULL,             -- e.g. 'Operation Epic Fury'
    event_timestamp TIMESTAMPTZ NOT NULL,      -- exact moment
    theme           TEXT NOT NULL,             -- e.g. 'Initial Outbreak'
    description     TEXT
);

-- Pre-defined temporal nodes from the paper
INSERT INTO temporal_nodes (node_index, label, event_timestamp, theme, description) VALUES
    (0,  'Operation Epic Fury',              '2026-02-27 00:00:00+00', 'Initial Outbreak',       'Pre-conflict military buildup and Operation announcement'),
    (1,  'Israeli-US Strikes',               '2026-02-28 00:00:00+00', 'Initial Outbreak',       'First kinetic action - Israeli and US strikes on Iran'),
    (2,  'Iranian Strikes',                  '2026-02-28 12:00:00+00', 'Initial Outbreak',       'Iranian retaliatory strikes; Supreme Leader death'),
    (3,  'Missiles toward British Bases',    '2026-03-01 00:00:00+00', 'Threshold Crossings',    'Two missiles fired toward British bases on Cyprus'),
    (4,  'Oil Refinery/Tanker Attacked',     '2026-03-01 12:00:00+00', 'Economic Shockwaves',    'Oil infrastructure targeted, energy market disruption'),
    (5,  'Qatar Halts Energy Production',    '2026-03-02 00:00:00+00', 'Economic Shockwaves',    'Qatar halts LNG production due to nearby attacks'),
    (6,  'Natanz Nuclear Facility Damaged',  '2026-03-02 12:00:00+00', 'Threshold Crossings',    'Strikes on Natanz nuclear site; leadership decapitation'),
    (7,  'US Suggests Citizen Evacuation',   '2026-03-03 00:00:00+00', 'Threshold Crossings',    'US officials suggest civilian evacuation from Middle East'),
    (8,  'Nine Countries; Ground Invasion',  '2026-03-03 12:00:00+00', 'Threshold Crossings',    'Nine countries involved; Israeli ground ops in Lebanon'),
    (9,  'Mojtaba Khamenei Becomes Leader',  '2026-03-03 18:00:00+00', 'Political Signaling',    'Succession crisis resolved; new Supreme Leader appointed'),
    (10, 'Late Escalation Node',             '2026-03-07 00:00:00+00', 'Political Signaling',    'Final observation node')
ON CONFLICT (node_index) DO NOTHING;

-- Questions (42 verifiable + 5 exploratory)
CREATE TABLE IF NOT EXISTS questions (
    id              SERIAL PRIMARY KEY,
    node_index      INTEGER REFERENCES temporal_nodes(node_index),  -- NULL for exploratory (asked at all nodes)
    question_text   TEXT NOT NULL,
    question_type   TEXT NOT NULL,  -- 'verifiable' or 'exploratory'
    theme           TEXT,
    ground_truth    BOOLEAN,        -- for verifiable: did event occur? NULL if unresolved
    notes           TEXT
);

-- LLM responses
CREATE TABLE IF NOT EXISTS llm_responses (
    id              SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL,          -- e.g. 'claude-sonnet-4.6', 'gpt-5.4'
    question_id     INTEGER REFERENCES questions(id),
    node_index      INTEGER REFERENCES temporal_nodes(node_index),
    context_article_count INTEGER,          -- how many articles in context
    context_char_count    INTEGER,          -- total chars in context
    full_response   TEXT,                   -- complete model output
    probability     REAL,                   -- extracted probability estimate (0-1)
    reasoning_tags  TEXT[],                 -- manually tagged reasoning patterns
    response_at     TIMESTAMPTZ DEFAULT NOW(),
    prompt_tokens   INTEGER,
    completion_tokens INTEGER
);

CREATE INDEX IF NOT EXISTS idx_responses_model ON llm_responses(model_name);
CREATE INDEX IF NOT EXISTS idx_responses_node ON llm_responses(node_index);
CREATE INDEX IF NOT EXISTS idx_responses_question ON llm_responses(question_id);

-- Context packages: pre-built article sets per temporal node
CREATE TABLE IF NOT EXISTS context_packages (
    id              SERIAL PRIMARY KEY,
    node_index      INTEGER REFERENCES temporal_nodes(node_index),
    article_id      INTEGER REFERENCES articles(id),
    position        INTEGER,  -- order in the context (1 = most recent)
    included_chars  INTEGER   -- how many chars of this article were included
);

CREATE INDEX IF NOT EXISTS idx_context_node ON context_packages(node_index);

-- View for quick corpus stats
CREATE OR REPLACE VIEW corpus_stats AS
SELECT 
    source_name,
    COUNT(*) as total_articles,
    COUNT(*) FILTER (WHERE scrape_status = 'success') as scraped,
    COUNT(*) FILTER (WHERE scrape_status = 'failed') as failed,
    COUNT(*) FILTER (WHERE scrape_status = 'pending') as pending,
    MIN(published_at) as earliest,
    MAX(published_at) as latest,
    ROUND(AVG(word_count) FILTER (WHERE word_count > 0)) as avg_words
FROM articles
GROUP BY source_name
ORDER BY total_articles DESC;
