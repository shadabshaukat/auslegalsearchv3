# AUSLegalSearch v3 — Post-load Indexing & Metadata Strategy (TB-scale)

This repository includes two production-ready SQL scripts for optimizing JSON metadata filtering, trigram search, and vector similarity over very large tables (multi‑TB). Choose the variant that fits your rollout constraints.

Contents
- schema-post-load/create_indexes.sql — Generated columns + simple indexes
- schema-post-load/create_indexes_expression.sql — Expression-index only (no table rewrite)
- tools/bench_sql_latency.py — End-to-end latency benchmark (p50/p95) for vector/FTS/metadata filters

Key metadata fields to surface/index (from ingestion JSON)
- Included: type, jurisdiction, subjurisdiction, database, date, year, title, titles[] (array), author, citations[] (array), citation (single), countries[] (array)
- Excluded (no indexing/columns): data_quality, url

Do I need extra data load?
- No re-ingestion or external data load is required.
- If you use GENERATED ALWAYS AS STORED columns (create_indexes.sql), PostgreSQL will materialize those columns from existing JSONB values. This causes a table rewrite for existing rows; schedule during a maintenance window for 4.5TB-scale tables.
- If you want to avoid a rewrite, use expression indexes only (create_indexes_expression.sql). These build over existing JSONB and do NOT rewrite the table.

Two rollout options

Option A: Generated Columns + Simple Indexes (create_indexes.sql)
- Adds md_* columns via GENERATED ALWAYS AS STORED for hot JSON keys.
- Builds straightforward btree/GIN/trigram indexes on these md_* columns.
- Pros: simpler query predicates, predictable planner behavior, easy future extensions.
- Cons: ALTER TABLE adds STORED columns → table rewrite. Use during maintenance.
- Requirements: PostgreSQL 12+ (for STORED), pg_trgm (for trigram), vector (for pgvector).

Option B: Expression Indexes Only (create_indexes_expression.sql)
- Builds expression indexes directly over JSONB (e.g., (chunk_metadata->>'type'), to_tsvector(...), lower(...)).
- Pros: no table rewrite; safe post-load; no schema changes.
- Cons: predicates must match index expressions; slightly more complex to maintain.
- Requirements: PostgreSQL, pg_trgm. No table rewrite.

Vector index choice (pgvector)
- Prefer HNSW if pgvector >= 0.7: lower tail latency for top‑k
  CREATE INDEX CONCURRENTLY idx_embeddings_vector_hnsw_cosine
    ON public.embeddings USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
  Query-time tuning:
  SET LOCAL hnsw.ef_search = 40..100; -- higher improves recall, increases latency

- If HNSW unavailable, use IVFFLAT with larger lists:
  DROP INDEX IF EXISTS idx_embeddings_vector_ivfflat_cosine;
  CREATE INDEX CONCURRENTLY idx_embeddings_vector_ivfflat_cosine
    ON public.embeddings USING ivfflat (vector vector_cosine_ops)
    WITH (lists = 4096);
  Query-time tuning:
  SET LOCAL ivfflat.probes = 8..16;

Session tuning for low latency
- Disable JIT for short queries:
  SET LOCAL jit = off;
- Set HNSW/IVF knobs per-query/route as shown above.

Post-index maintenance
- After building indexes or bulk loads:
  ANALYZE public.embeddings;
  ANALYZE public.documents;

Queries and predicates (write predicates to match the indexes)

If using generated columns (create_indexes.sql), prefer md_* columns where possible:
- Equality/range
  WHERE e.md_type = 'case'
    AND e.md_jurisdiction = 'cth'
    AND e.md_database = 'HCA'
    AND e.md_year = 1927
    AND e.md_date BETWEEN '1927-01-01' AND '1927-12-31'

- Array membership
  WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
    WHERE LOWER(c.val) = LOWER('United States')
  )

- Trigram (approximate)
  WHERE e.md_title_lc % LOWER(:title_approx)
  -- or for author
  WHERE e.md_author_lc % LOWER(:author_approx)

- documents.source trigram (approximate)
  WHERE EXISTS (
    SELECT 1
    FROM documents d
    WHERE d.id = e.doc_id
      AND d.source_lc % LOWER(:src_approx)
  )

If using expression indexes only (create_indexes_expression.sql), ensure predicates match expressions:
- Equality/range
  WHERE (e.chunk_metadata->>'type') = 'case'
    AND (e.chunk_metadata->>'jurisdiction') = 'cth'
    AND (e.chunk_metadata->>'database') = 'HCA'
    AND ((e.chunk_metadata->>'year')::int) = 1927
    AND ((e.chunk_metadata->>'date')::date) BETWEEN '1927-01-01' AND '1927-12-31'

- Array membership
  WHERE EXISTS (
    SELECT 1
    FROM jsonb_array_elements_text(e.chunk_metadata->'countries') AS c(val)
    WHERE LOWER(c.val) = LOWER('United States')
  )

- Trigram (approximate)
  WHERE LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx)

- documents.source trigram (approximate)
  WHERE LOWER(d.source) % LOWER(:src_approx)

End-to-end latency benchmark (p50/p95)
- Use tools/bench_sql_latency.py. It measures:
  - Vector (pgvector + JSON filters)
  - FTS (documents_fts)
  - Metadata-only filters
  - Hybrid combine client-side
- Examples:
  python3 tools/bench_sql_latency.py \
    --query "Australia Peru investment agreement" \
    --top_k 10 --runs 10 --probes 10 --hnsw_ef 60 \
    --type treaty --subjurisdiction dfat --jurisdiction au --database ATS \
    --country "United States"

  python3 tools/bench_sql_latency.py \
    --query "Angelides v James Stedman Hendersons" \
    --top_k 10 --runs 10 --probes 12 \
    --type case --jurisdiction cth --database HCA \
    --title_member "Angelides v James Stedman Hendersons Sweets Ltd" \
    --citation_member "[1927] HCA 34"

Operational guidance for TB-scale
- If you cannot afford a table rewrite now:
  - Apply expression indexes first (create_indexes_expression.sql).
  - Validate with bench_sql_latency.py to reach <50ms p95.
  - Plan a maintenance window later if you want to migrate to generated columns.

- If you can accept a rewrite window:
  - Apply create_indexes.sql once. Expect table rewrite time proportional to size.
  - Build HNSW (or IVFFLAT lists) in parallel and ANALYZE.
  - Validate with the benchmark; tune probes/ef_search.

About “dbtools” syntax errors in editor
- Some IDE DB linters don’t recognize PostgreSQL JSONB operators (->, ->>) or GENERATED columns, and will flag false errors.
- These scripts are valid for PostgreSQL. Execute them via psql or a Postgres-aware admin tool. If your tool still rejects expressions, use the generated-columns script or psql.

Contact points for further tuning
- Provide EXPLAIN (ANALYZE, BUFFERS) for slow queries and the output of:
  SELECT extversion FROM pg_extension WHERE extname='vector';
- We can adjust partial indexes, lists/ef_search, or query shapes further based on real plans.
