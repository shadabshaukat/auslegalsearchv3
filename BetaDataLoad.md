# AUSLegalSearch v3 — Beta Data Load Guide

End-to-end guide to run the beta ingestion pipeline: discover files, parse, chunk (semantic or dashed-legislation), embed on NVIDIA GPUs, write to Postgres/pgvector, and maintain FTS. Includes single-worker and multi-GPU orchestration, resume, performance tuning, and verification.

Contents
- 1) Overview
- 2) Prerequisites
- 3) Directory layout and supported files
- 4) Multi-GPU full ingest (recommended)
- 5) Sample/preview ingest
- 6) Single-worker runs (advanced)
- 7) Resume and checkpointing
- 8) Chunking mechanics map (beta workflow)
- 9) Configuration flags and environment variables
- 10) Logs, monitoring, and verification
- 11) Performance tuning on NVIDIA GPUs
- 12) Troubleshooting


## 1) Overview

Pipeline steps per file:
1. Parse: .txt/.html to text and file-level metadata.
2. Chunk:
   - Primary: semantic token-aware chunking (headings → sentences; token budgets).
   - Legislation-aware: dashed header blocks (title/regulation/chunk_id) chunked section-by-section semantically; metadata propagated into chunk_metadata.
   - Fallback: character-window chunker on chunk failures/timeouts; full coverage guaranteed.
3. Embed: HuggingFace embeddings on GPU (batched).
4. Persist: write to Postgres tables documents + embeddings (pgvector) and maintain FTS via trigger.
5. Track: record EmbeddingSession and EmbeddingSessionFile for status, safe resume, and logs.

Key modules:
- Orchestrator (multi-GPU): ingest/beta_orchestrator.py
- Worker (per GPU): ingest/beta_worker.py
- Chunkers: ingest/semantic_chunker.py (beta), ingest/loader.py (legacy)
- Embedding: embedding/embedder.py
- DB and FTS: db/store.py, db/connector.py


## 2) Prerequisites

- Python 3.10+ virtualenv with project requirements installed:
  - pip install -r requirements.txt
- Postgres with pgvector and extensions enabled.
  - db/store.py create_all_tables() ensures:
    - CREATE EXTENSION IF NOT EXISTS vector, pg_trgm, uuid-ossp, fuzzystrmatch
    - FTS: documents.document_fts, trigger function, GIN index
    - IVFFLAT/other indexes for vector search
- Database env vars (example):
  - export AUSLEGALSEARCH_DB_HOST=localhost
  - export AUSLEGALSEARCH_DB_PORT=5432
  - export AUSLEGALSEARCH_DB_USER=postgres
  - export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
  - export AUSLEGALSEARCH_DB_NAME=postgres
- NVIDIA GPUs and CUDA visible for embedding acceleration:
  - nvidia-smi available
  - Optional: set HF_HOME on fast disk for model caching.

Environment loading from .env (important)
- The code reads configuration from environment variables via os.environ at runtime and now also auto-loads a .env file at import time (db/connector.py) from either:
  - repo root: ./.env
  - current working directory: ./.env
- Exported environment variables always take precedence over .env file values. If a key is already exported, the .env loader will not override it.

Optional: export all .env variables in the shell (useful for other tools too):
```
set -a
source .env
set +a
# verify:
env | grep AUSLEGALSEARCH_DB_
python3 -c 'import os;print(os.environ.get("AUSLEGALSEARCH_DB_PASSWORD"))'
```
Notes:
- Because the code now auto-loads .env, simply running python from the repo root or CWD that contains .env is sufficient. Exporting variables is still fine and overrides .env.
- Alternatively, set a single URL:
  - AUSLEGALSEARCH_DB_URL='postgresql+psycopg2://user:pass@host:5432/dbname'
  - If password contains special characters (@ : / ? # & + %), percent-encode them. (The code will URL-encode user/password when building from separate vars, but AUSLEGALSEARCH_DB_URL takes precedence if set.)
- Another option is to use a dotenv runner (if you install python-dotenv):
  - `python -m dotenv -f .env run -- python3 -m ingest.beta_orchestrator ...`


## 3) Directory layout and supported files

- Data root: directory tree with .txt/.html files (UTF-8).
- Supported extensions: .txt, .html
- Year directories are NOT skipped by the full ingest (beta); sample mode can skip them for previews.
- Legislation with dashed headers (preferred format):
  -----------------------------------
  title: Fee for providing ATAGI advice
  regulation: 7
  chunk_id: 0
  -----------------------------------
  Body text...
  ... until next dashed header or EOF ...


## 4) Multi-GPU full ingest (recommended)

The orchestrator auto-detects GPUs and launches one worker per GPU, partitioning files evenly. Child sessions are created as {base_session}-gpu0, -gpu1, ...

Basic full ingest (auto GPU detect):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --log_dir "/abs/path/to/logs"
```

Specify embedding model (HuggingFace):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Explicitly set GPU count:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-2gpu" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Token-aware chunk sizes (cascade to all beta chunkers, including dashed-legislation):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-acts-1500tok" \
  --target_tokens 1500 \
  --overlap_tokens 192 \
  --max_tokens 1920 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --log_dir "/abs/path/to/logs"
```

Launch without waiting (fire-and-forget, no log aggregation):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-nowait" \
  --no_wait \
  --log_dir "/abs/path/to/logs"
```


## 5) Sample/preview ingest

Pick one file per folder (skipping year directories by default) for quick dry runs:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sample" \
  --sample_per_folder \
  --log_dir "/abs/path/to/logs"
```

Do not skip year directories in sample mode:
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-sample-all" \
  --sample_per_folder \
  --no_skip_years_in_sample \
  --log_dir "/abs/path/to/logs"
```


## 6) Single-worker runs (advanced)

Directly run a worker (e.g., to test one GPU or a single partition). You control CUDA_VISIBLE_DEVICES:

Single worker over entire root:
```
CUDA_VISIBLE_DEVICES=0 \
python3 -m ingest.beta_worker child-session-gpu0 \
  --root "/path/to/Data_for_Beta_Launch" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Single worker from a partition file:
```
# Partition file contains absolute file paths, one per line
CUDA_VISIBLE_DEVICES=1 \
python3 -m ingest.beta_worker child-session-gpu1 \
  --partition_file ".beta-gpu-partition-child-session-gpu1.txt" \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```

Flags (beta_worker):
- Positional: session_name
- --root or --partition_file (one is required)
- --model
- --target_tokens, --overlap_tokens, --max_tokens
- --log_dir


## 7) Resume and checkpointing

Resume is supported at child-session granularity. Files already marked complete for the SAME child session are skipped.

How it works:
- Each worker records EmbeddingSessionFile rows keyed by (session_name, filepath) with status:
  - pending → complete or error
- Re-running the same session_name will skip all files with status complete.

Resuming a multi-GPU run:
- Reuse the same base session name and the same GPU count (so child session names match, e.g., beta-full-...-gpu0, -gpu1, ...):
```
python3 -m ingest.beta_orchestrator \
  --root "/path/to/Data_for_Beta_Launch" \
  --session "beta-full-20250101-120000" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920 \
  --log_dir "/abs/path/to/logs"
```
- The workers will log SKIP already complete ... for processed files.

Important notes:
- Changing GPU count changes child session names. Current skip logic does not dedupe across different child sessions; use the same GPU count to resume reliably.
- Partition files are regenerated; with the same inputs/GPU count, partitions remain consistent.

Single-worker resume:
- Reuse the same session_name:
```
CUDA_VISIBLE_DEVICES=0 \
python3 -m ingest.beta_worker child-session-gpu0 \
  --root "/path/to/Data_for_Beta_Launch" \
  --log_dir "/abs/path/to/logs"
```
- Already completed files for child-session-gpu0 are skipped.


## 8) Chunking mechanics map (beta workflow)

This section explains how text is chunked before embedding, and how metadata is propagated to support retrieval and explainability.

A) Legislation dashed-header path (new beta workflow)
- Trigger: attempted first for every file; if dashed blocks are found.
- Marker format:
  -----------------------------------
  key: value
  key: value
  key: value
  -----------------------------------
  Body text (until next dashed header or EOF)
- Parser:
  - ingest/semantic_chunker.parse_dashed_blocks identifies all dashed headers and their bodies.
  - Each header is parsed into header_meta dict (keys lowercased), e.g., title, regulation, chunk_id.
- Semantic chunking per block:
  - ingest/semantic_chunker.chunk_legislation_dashed_semantic(body, base_meta, cfg)
  - Token-aware: ChunkingConfig(target_tokens, overlap_tokens, max_tokens)
  - Sentences merged to target_tokens with overlap; never exceed max_tokens (long sentences hard-split).
- Metadata per chunk:
  - chunk_metadata = base_meta + header_meta + per-chunk meta
  - Includes:
    - title, regulation, chunk_id
    - section_title (title), section_idx, chunk_idx, tokens_est
    - path-derived fields (jurisdiction_guess, rel_path, etc.)
- Preface handling:
  - If there is text before the first dashed header (e.g., when the loader stripped the first header to file-level metadata), that text is also semantic-chunked and included (with base_meta but without header_meta).

B) Generic semantic chunker (fallback when no dashed blocks found)
- ingest/semantic_chunker.chunk_document_semantic(text, base_meta, cfg):
  - split_into_blocks(text): detect headings (Roman numerals, 1./1.2.3., UPPERCASE, “Section N” or “s. N”).
  - chunk_text_semantic per block: sentence-based with token targets and overlap.
- Metadata per chunk:
  - chunk_metadata = base_meta + {"section_title", "section_idx", "chunk_idx", "tokens_est"}

C) Fallback character-window chunker (on timeouts/errors in chunk stage)
- Behavior:
  - Slices full text into fixed-size overlapping windows until the end; guarantees 100% coverage.
- Controls:
  - AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK (default 4000)
  - AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS (default 200)
  - step = max(1, chunk_size - overlap)
- Metadata marks:
  - strategy: "fallback-naive"
  - fallback: True
  - start_char, end_char

Legacy loader-based chunking (unchanged, for reference)
- ingest/loader.py has chunk_legislation(), chunk_case(), chunk_journal(), etc., with a 1500-character chunk policy.
- Beta worker uses the semantic chunkers described above; legacy loader chunking is left intact for older pipelines.

D) Cases (beta semantic strategy)
- Detection: detect_doc_type(meta, text) classifies “case” using heuristics (e.g., presence of “ v ”, “appellant”, “respondent”).
- Strategy: the beta pipeline uses the generic semantic chunker (chunk_document_semantic):
  - split_into_blocks detects legal-style headings and numbered paragraphs.
  - chunk_text_semantic merges sentences to ~target_tokens with overlap; never exceeds max_tokens (long sentences hard-split).
- Metadata per chunk: chunk_metadata includes section_title (heading text if any), section_idx, chunk_idx, tokens_est, plus path/file metadata and detected type when available.
- Legacy (unchanged): loader.chunk_case() packs paragraphs to ≤1500 characters and falls back as needed. The beta path does not call this legacy function.

E) Journals (beta semantic strategy)
- Detection: detect_doc_type identifies “journal” primarily via structured headings (Roman numerals, 1., 1.2.3., single-letter sections).
- Strategy: uses the same generic semantic chunker as above; headings create blocks; sentences are merged to target_tokens with overlap; max_tokens enforced.
- Metadata per chunk: mirrors cases; section_title is the detected heading; tokens_est, chunk_idx, section_idx are set; path/file metadata is included.
- Legacy (unchanged): loader.chunk_journal() groups by detected headings and applies a 1500-character policy. The beta path does not call this legacy function.


## 9) Configuration flags and environment variables

Orchestrator flags (ingest/beta_orchestrator.py):
- --root (required): dataset root directory
- --session (required): base session name; child sessions are {session}-gpu0, {session}-gpu1, ...
- --model: embedding model (e.g., "nomic-ai/nomic-embed-text-v1.5")
- --gpus: number of GPUs (0 = auto-detect)
- --sample_per_folder: pick one file per folder (preview mode)
- --no_skip_years_in_sample: in sample mode, also include year directories
- --target_tokens: token target per chunk (default 512)
- --overlap_tokens: overlapping tokens between chunks (default 64)
- --max_tokens: hard max per chunk (default 640)
- --log_dir: logs directory
- --no_wait: do not wait (no aggregation at end)

Worker flags (ingest/beta_worker.py):
- Positional: session_name (e.g., "beta-...-gpu0")
- --root or --partition_file
- --model
- --target_tokens / --overlap_tokens / --max_tokens
- --log_dir

Environment variables:
- Database:
  - AUSLEGALSEARCH_DB_HOST / AUSLEGALSEARCH_DB_PORT / AUSLEGALSEARCH_DB_USER / AUSLEGALSEARCH_DB_PASSWORD / AUSLEGALSEARCH_DB_NAME
  - AUSLEGALSEARCH_DB_URL (optional override): full SQLAlchemy URL, e.g.:
    - postgresql+psycopg2://user:pass@host:5432/dbname
    - If your password contains special characters (@ : / ? #), prefer AUSLEGALSEARCH_DB_URL or percent-encode them (e.g., @ -> %40). The code now URL-encodes user/password automatically when building from the individual vars, but AUSLEGALSEARCH_DB_URL takes precedence if set.
- Embedding:
  - AUSLEGALSEARCH_EMBED_BATCH (default 64) — increase for throughput if GPU memory allows.
- Timeouts (seconds):
  - AUSLEGALSEARCH_TIMEOUT_PARSE (default 60)
  - AUSLEGALSEARCH_TIMEOUT_CHUNK (default 90)
  - AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH (default 180)
- Fallback:
  - AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT (default "1" enabled)
  - AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK (default 4000)
  - AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS (default 200)
- Diagnostics:
  - AUSLEGALSEARCH_ERROR_DETAILS (default "1")
  - AUSLEGALSEARCH_ERROR_TRACE (default "0")
  - AUSLEGALSEARCH_DEBUG_COUNTS (default "0")
- CUDA/Transformers (optional):
  - TOKENIZER_PARALLELISM=0
  - HF_HOME=/fast/ssd/hf_cache
  - TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


## 10) Logs, monitoring, and verification

Worker per-child logs (incremental appends):
- {child_session}.success.log
- {child_session}.error.log
- {child_session}.errors.ndjson (structured; only if AUSLEGALSEARCH_ERROR_DETAILS=1)

Orchestrator aggregated logs (when wait enabled):
- {base_session}.success.log
- {base_session}.error.log

Console output includes DB target, log paths, and periodic progress.

Quick DB sanity checks (psql examples):
```
-- counts
SELECT count(*) FROM documents;
SELECT count(*) FROM embeddings;

-- sample chunks with dashed legislation metadata
SELECT id,
       chunk_metadata->>'title' AS title,
       chunk_metadata->>'regulation' AS regulation,
       chunk_metadata->>'chunk_id' AS chunk_id,
       (chunk_metadata->>'tokens_est')::int AS tokens_est
FROM embeddings
WHERE chunk_metadata ? 'title'
ORDER BY id DESC
LIMIT 10;

-- FTS search check
SELECT id, source
FROM documents
WHERE document_fts @@ plainto_tsquery('english', 'pharmaceutical')
LIMIT 20;

-- Vector search requires functions in app layer; check indexes/ops
\d+ embeddings
```

Resume visibility:
- On resume, worker logs output lines like:
  [beta_worker] {child_session}: SKIP already complete {i}/{N} -> /abs/filepath


## 11) Performance tuning on NVIDIA GPUs

- CPU parallelization:
  - Parallelism model: ingest/beta_orchestrator launches one worker process per GPU. Each worker performs parsing and semantic chunking on CPU and embedding on the assigned GPU.
  - Within a worker: chunking is single-process, single-threaded Python (regex/sentence operations). There is no per-file multiprocessing inside a worker.
  - If legislation chunking is the bottleneck on a single-GPU host:
    - Use more GPUs (—gpus N) to increase the number of workers and parallelize chunking across processes.
    - Advanced (manual) option: launch additional beta_worker processes bound to the same GPU (same CUDA_VISIBLE_DEVICES) to parallelize CPU chunking; lower AUSLEGALSEARCH_EMBED_BATCH to avoid GPU OOM due to concurrent embeddings. This is not orchestrator-managed; use with care.
    - Increase target_tokens to reduce the number of chunks per file (fewer embed calls).
    - Ensure fast CPU and disk; the semantic tokenizer is CPU-bound and benefits from high single-core performance.
  - Tokenizers note: this pipeline uses lightweight regex tokenization (not HuggingFace tokenizers) for chunk sizing; TOKENIZER_PARALLELISM generally has no effect on the chunking step.

- Workers per GPU:
  - Orchestrator launches one worker per detected GPU (or per --gpus). Ensure GPUs are healthy via nvidia-smi.
- Embedding batch size:
  - AUSLEGALSEARCH_EMBED_BATCH: increase for throughput (watch memory).
    - 16 GB GPUs: 64–128
    - 24–40 GB GPUs: 128–256
- Chunk token sizes:
  - Larger target_tokens produce fewer chunks and can improve throughput; balance with retrieval quality.
  - Example for legislation: --target_tokens 1500 --overlap_tokens 192 --max_tokens 1920
- Model caching:
  - Set HF_HOME to a fast SSD; warm the model before large runs.
- Avoid frequent fallbacks:
  - Increase AUSLEGALSEARCH_TIMEOUT_CHUNK moderately for very long Acts if you see frequent chunk timeouts.
- CPU considerations:
  - Semantic chunking is CPU-bound but relatively light; parallel workers distribute load across CPU cores.


## 12) Troubleshooting

- No files found:
  - Verify --root path and supported extensions (.txt/.html).
- Database errors on vector/FTS:
  - Ensure db/store.py create_all_tables() ran at least once with a superuser if extensions require it.
- Embedding out-of-memory (OOM):
  - Reduce AUSLEGALSEARCH_EMBED_BATCH.
  - Lower token sizes to reduce text per batch.
- Slow downloads / repeated model fetches:
  - Set HF_HOME to a persistent cache directory.
- Resume didn’t skip:
  - Confirm you reused the SAME base session and SAME GPU count (child session names must match).
  - Check EmbeddingSessionFile rows for the child session; files should have status 'complete'.
- Too many fallback chunks:
  - Increase AUSLEGALSEARCH_TIMEOUT_CHUNK if semantic chunking times out.
  - Ensure overlap in fallback is < chunk size (defaults are safe: 4000/200).
- Verify dashed-legislation metadata:
  - Query embeddings where chunk_metadata ? 'title' and inspect regulation/chunk_id fields.

Example end-to-end command (production-style):
```
export AUSLEGALSEARCH_DB_HOST=localhost
export AUSLEGALSEARCH_DB_PORT=5432
export AUSLEGALSEARCH_DB_USER=postgres
export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
export AUSLEGALSEARCH_DB_NAME=postgres
export AUSLEGALSEARCH_EMBED_BATCH=128
export TOKENIZER_PARALLELISM=0
export HF_HOME=/fast/ssd/hf_cache

python3 -m ingest.beta_orchestrator \
  --root "/home/ubuntu/Data_for_Beta_Launch" \
  --session "beta-full-$(date +%Y%m%d-%H%M%S)" \
  --gpus 2 \
  --model "nomic-ai/nomic-embed-text-v1.5" \
  --target_tokens 1500 \
  --overlap_tokens 192 \
  --max_tokens 1920 \
  --log_dir "/home/ubuntu/auslegalsearchv3/auslegalsearchv3/logs"
```

This guide reflects the current beta workflow and code paths:
- Orchestrator: ingest/beta_orchestrator.py
- Worker: ingest/beta_worker.py
- Chunkers (beta): ingest/semantic_chunker.py (dashed-legislation and generic semantic)
- Embedding: embedding/embedder.py
- DB/FTS: db/store.py, db/connector.py
