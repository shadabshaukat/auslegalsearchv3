"""
Parallel beta ingestion worker for AUSLegalSearch v3.

This worker:
- Processes a provided list of files (via --partition_file) OR discovers all files under --root.
- Parses (.txt/.html), performs semantic, token-aware chunking, embeds (batched), and writes to DB.
- Tracks progress in EmbeddingSession and EmbeddingSessionFile for the given session_name.
- Intended to be launched by a multi-GPU orchestrator, one process per GPU with CUDA_VISIBLE_DEVICES set.
- Writes per-worker log files listing successfully ingested files and failures.
- NEW: Incremental logging (append to logs per file) and batched embedding to avoid memory stalls.

Usage (typically invoked by orchestrator):
    python -m ingest.beta_worker SESSION_NAME \\
        --root "/abs/path/to/Data_for_Beta_Launch" \\
        --partition_file ".beta-gpu-partition-SESSION_NAME-gpu0.txt" \\
        --model "nomic-ai/nomic-embed-text-v1.5" \\
        --target_tokens 512 --overlap_tokens 64 --max_tokens 640 \\
        --log_dir "./logs"
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy import text

import signal
import contextlib
import json
import traceback

from ingest.loader import parse_txt, parse_html
from ingest.semantic_chunker import chunk_document_semantic, ChunkingConfig, detect_doc_type, chunk_legislation_dashed_semantic, chunk_generic_rcts
from embedding.embedder import Embedder

from db.store import (
    create_all_tables,
    start_session, update_session_progress, complete_session, fail_session,
    EmbeddingSessionFile, SessionLocal, Document, Embedding
)
from db.connector import DB_URL

# Timeouts (seconds). Tunable via env.
PARSE_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_PARSE", "60"))
CHUNK_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_CHUNK", "90"))
EMBED_BATCH_TIMEOUT = int(os.environ.get("AUSLEGALSEARCH_TIMEOUT_EMBED_BATCH", "180"))

class _Timeout(Exception):
    pass

@contextlib.contextmanager
def _deadline(seconds: int):
    if seconds is None or seconds <= 0:
        yield
        return
    def _handler(signum, frame):
        raise _Timeout(f"operation exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

SUPPORTED_EXTS = {".txt", ".html"}
_YEAR_DIR_RE = re.compile(r"^(19|20)\d{2}$")  # 1900-2099


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s or "")]


def find_all_supported_files(root_dir: str) -> List[str]:
    """Recursively list ALL supported files under root_dir. No skipping of year directories."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = sorted(dirnames, key=_natural_sort_key)
        files = [f for f in filenames if Path(f).suffix.lower() in SUPPORTED_EXTS]
        for f in sorted(files, key=_natural_sort_key):
            out.append(os.path.abspath(os.path.join(dirpath, f)))
    # dedupe + sort
    return sorted(list(dict.fromkeys(out)), key=_natural_sort_key)


def read_partition_file(fname: str) -> List[str]:
    with open(fname, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def derive_path_metadata(file_path: str, root_dir: str) -> Dict[str, Any]:
    """
    Derive helpful metadata from the folder structure.
    """
    root_dir = os.path.abspath(root_dir) if root_dir else ""
    file_path = os.path.abspath(file_path)
    rel_path = os.path.relpath(file_path, root_dir) if root_dir and file_path.startswith(root_dir) else file_path
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p]

    parts_no_years = [p for p in parts if not _YEAR_DIR_RE.match(p or "")]
    jurisdiction_guess = parts_no_years[0].lower() if parts_no_years else None

    court_guess = None
    if parts_no_years:
        if len(parts_no_years) >= 2:
            last_non_year = parts_no_years[-2] if "." in parts_no_years[-1] else parts_no_years[-1]
            court_guess = last_non_year

    series_guess = None
    if len(parts_no_years) >= 3:
        series_guess = "/".join(parts_no_years[1:3]).lower()

    return {
        "dataset_root": root_dir,
        "rel_path": rel_path,
        "rel_path_no_years": "/".join(parts_no_years),
        "path_parts": parts,
        "path_parts_no_years": parts_no_years,
        "jurisdiction_guess": jurisdiction_guess,
        "court_guess": court_guess,
        "series_guess": series_guess,
        "filename": os.path.basename(file_path),
        "ext": Path(file_path).suffix.lower(),
    }


def parse_file(filepath: str) -> Dict[str, Any]:
    ext = Path(filepath).suffix.lower()
    if ext == ".txt":
        return parse_txt(filepath)
    if ext == ".html":
        return parse_html(filepath)
    return {}


def _batch_insert_chunks(
    session,
    chunks: List[Dict[str, Any]],
    vectors,
    source_path: str,
    fmt: str
) -> int:
    """
    Insert a batch of chunks and matching vectors into the database as
    Document and Embedding rows. Returns inserted count.
    """
    if not chunks:
        return 0
    if vectors is None or len(vectors) != len(chunks):
        raise ValueError("Vector batch does not match number of chunks")

    inserted = 0
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        cm = chunk.get("chunk_metadata") or {}
        doc = Document(
            source=source_path,
            content=text,
            format=fmt.strip(".") if fmt.startswith(".") else fmt
        )
        session.add(doc)
        session.flush()  # assign id
        emb = Embedding(
            doc_id=doc.id,
            chunk_index=idx,
            vector=vectors[idx],
            chunk_metadata=cm
        )
        session.add(emb)
        inserted += 1
    session.commit()
    return inserted


def _append_log_line(log_dir: str, session_name: str, file_path: str, success: bool) -> None:
    os.makedirs(log_dir, exist_ok=True)
    fname = f"{session_name}.success.log" if success else f"{session_name}.error.log"
    fpath = os.path.join(log_dir, fname)
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")


def _metrics_enabled() -> bool:
    return os.environ.get("AUSLEGALSEARCH_LOG_METRICS", "1") != "0"


def _append_success_metrics_line(
    log_dir: str,
    session_name: str,
    file_path: str,
    chunks_count: int,
    text_len: int,
    cfg: ChunkingConfig,
    strategy: str,
    detected_type: Optional[str] = None,
    section_count: Optional[int] = None,
    tokens_est_total: Optional[int] = None,
    tokens_est_mean: Optional[int] = None,
    parse_ms: Optional[int] = None,
    chunk_ms: Optional[int] = None,
    embed_ms: Optional[int] = None,
    insert_ms: Optional[int] = None,
) -> None:
    """
    Append a single TSV-style line to the child success log with per-file metrics.
    Keeps logging overhead minimal for performance.
    """
    try:
        os.makedirs(log_dir, exist_ok=True)
        fpath = os.path.join(log_dir, f"{session_name}.success.log")
        if not _metrics_enabled():
            # Write only the file path (legacy format) and avoid computing metrics
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(str(file_path) + "\n")
            return
        parts = [
            str(file_path),
            f"chunks={int(chunks_count)}",
            f"text_len={int(text_len)}",
            f"strategy={strategy or ''}",
            f"target_tokens={int(cfg.target_tokens)}",
            f"overlap_tokens={int(cfg.overlap_tokens)}",
            f"max_tokens={int(cfg.max_tokens)}",
        ]
        if detected_type:
            parts.append(f"type={detected_type}")
        if section_count is not None:
            parts.append(f"section_count={int(section_count)}")
        if tokens_est_total is not None:
            parts.append(f"tokens_est_total={int(tokens_est_total)}")
        if tokens_est_mean is not None:
            parts.append(f"tokens_est_mean={int(tokens_est_mean)}")
        if parse_ms is not None:
            parts.append(f"parse_ms={int(parse_ms)}")
        if chunk_ms is not None:
            parts.append(f"chunk_ms={int(chunk_ms)}")
        if embed_ms is not None:
            parts.append(f"embed_ms={int(embed_ms)}")
        if insert_ms is not None:
            parts.append(f"insert_ms={int(insert_ms)}")
        line = "\t".join(parts)
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Never fail pipeline due to logging
        pass


def _write_logs(log_dir: str, session_name: str, successes: List[str], failures: List[str]) -> Dict[str, str]:
    """
    Finalize logs. We DO NOT overwrite the child success log to preserve
    the per-file metrics lines emitted during processing. Instead, append a compact
    summary footer to each log. This keeps overhead negligible.
    """
    os.makedirs(log_dir, exist_ok=True)
    succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    fail_path = os.path.join(log_dir, f"{session_name}.error.log")

    # Ensure files exist
    try:
        open(succ_path, "a", encoding="utf-8").close()
    except Exception:
        pass
    try:
        open(fail_path, "a", encoding="utf-8").close()
    except Exception:
        pass

    # Deduplicate file path lists (for counts only; we don't rewrite logs)
    successes = list(dict.fromkeys(successes))
    failures = list(dict.fromkeys(failures))

    # Append summary footers (minimal I/O)
    try:
        with open(succ_path, "a", encoding="utf-8") as f:
            f.write(f"# summary files_ok={len(successes)}\n")
    except Exception:
        pass
    try:
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(f"# summary files_failed={len(failures)}\n")
    except Exception:
        pass

    return {"success_log": succ_path, "error_log": fail_path}

def _error_details_enabled() -> bool:
    return os.environ.get("AUSLEGALSEARCH_ERROR_DETAILS", "1") == "1"

def _append_error_detail(
    log_dir: str,
    session_name: str,
    file_path: str,
    stage: str,
    error_type: str,
    message: str,
    duration_ms: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
    tb: Optional[str] = None
) -> None:
    if not _error_details_enabled():
        return
    try:
        rec = {
            "session": session_name,
            "file": file_path,
            "stage": stage,
            "error_type": error_type,
            "message": message,
            "duration_ms": duration_ms,
            "meta": meta or {},
            "ts": int(time.time() * 1000),
        }
        if os.environ.get("AUSLEGALSEARCH_ERROR_TRACE", "0") == "1":
            rec["traceback"] = tb or ""
        path = os.path.join(log_dir, f"{session_name}.errors.ndjson")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # Never fail pipeline due to logging
        pass


def _maybe_print_counts(dbs, session_name: str, label: str) -> None:
    """
    If AUSLEGALSEARCH_DEBUG_COUNTS=1, print current documents/embeddings counts with a label.
    """
    try:
        if os.environ.get("AUSLEGALSEARCH_DEBUG_COUNTS", "0") != "1":
            return
        docs = dbs.execute(text("SELECT count(*) FROM documents")).scalar()
        embs = dbs.execute(text("SELECT count(*) FROM embeddings")).scalar()
        print(f"[beta_worker] {session_name}: DB counts {label}: documents={docs}, embeddings={embs}", flush=True)
    except Exception as e:
        print(f"[beta_worker] {session_name}: DB counts {label}: error: {e}", flush=True)


def _embed_in_batches(embedder: Embedder, texts: List[str], batch_size: int) -> List:
    """
    Embed texts in smaller batches to avoid large memory spikes / stalls.
    Returns a list-like of vectors aligned with texts.
    """
    if batch_size <= 0:
        batch_size = 64
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        sub = texts[i:i + batch_size]
        with _deadline(EMBED_BATCH_TIMEOUT):
            vecs = embedder.embed(sub)  # ndarray [batch, dim]
        # Extend list with each row (so downstream indexing works)
        for j in range(vecs.shape[0]):
            all_vecs.append(vecs[j])
    return all_vecs

def _fallback_chunk_text(text: str, base_meta: Dict[str, Any], cfg: ChunkingConfig) -> List[Dict[str, Any]]:
    """
    Fallback chunker that slices text by characters when semantic chunker times out or fails.
    Uses configurable window/overlap to create chunk dicts compatible with downstream pipeline.
    """
    chars_per_chunk = int(os.environ.get("AUSLEGALSEARCH_FALLBACK_CHARS_PER_CHUNK", "4000"))
    overlap_chars = int(os.environ.get("AUSLEGALSEARCH_FALLBACK_OVERLAP_CHARS", "200"))
    if chars_per_chunk <= 0:
        chars_per_chunk = 4000
    if overlap_chars < 0:
        overlap_chars = 0
    step = max(1, chars_per_chunk - overlap_chars)
    chunks: List[Dict[str, Any]] = []
    n = len(text or "")
    i = 0
    idx = 0
    while i < n:
        j = min(n, i + chars_per_chunk)
        slice_text = text[i:j]
        md = dict(base_meta or {})
        md.setdefault("strategy", "fallback-naive")
        md["fallback"] = True
        md["start_char"] = i
        md["end_char"] = j
        chunks.append({"text": slice_text, "chunk_metadata": md})
        idx += 1
        i += step
    return chunks


def run_worker(
    session_name: str,
    root_dir: Optional[str],
    partition_file: Optional[str],
    embedding_model: Optional[str],
    token_target: int,
    token_overlap: int,
    token_max: int,
    log_dir: str
) -> None:
    """
    Main loop for this worker.
    """
    create_all_tables()  # Ensure extensions, indexes, triggers

    # Resolve file list
    if partition_file:
        files = read_partition_file(partition_file)
    else:
        if not root_dir:
            raise ValueError("Either --partition_file or --root must be provided")
        files = find_all_supported_files(root_dir)

    embedder = Embedder(embedding_model) if embedding_model else Embedder()
    cfg = ChunkingConfig(
        target_tokens=int(token_target),
        overlap_tokens=int(token_overlap),
        max_tokens=int(token_max),
    )

    batch_size = int(os.environ.get("AUSLEGALSEARCH_EMBED_BATCH", "64"))

    processed_chunks = 0
    successes: List[str] = []
    failures: List[str] = []

    total_files = len(files)
    # Ensure log dir exists; print DB target and log file paths for diagnostics
    os.makedirs(log_dir, exist_ok=True)
    try:
        from urllib.parse import urlparse as _urlparse
        _p = _urlparse(DB_URL)
        _safe_db = f"{_p.scheme}://{_p.hostname}:{_p.port}/{_p.path.lstrip('/')}"
    except Exception:
        _safe_db = DB_URL
    _succ_path = os.path.join(log_dir, f"{session_name}.success.log")
    _err_path = os.path.join(log_dir, f"{session_name}.error.log")
    print(f"[beta_worker] {session_name}: DB = {_safe_db}", flush=True)
    print(f"[beta_worker] {session_name}: logs: success->{_succ_path}, error->{_err_path}", flush=True)
    print(f"[beta_worker] {session_name}: starting. files={total_files}, batch_size={batch_size}", flush=True)

    with SessionLocal() as dbs:
        current_stage: Optional[str] = None
        stage_start: float = 0.0
        current_text_len: Optional[int] = None
        current_chunk_count: Optional[int] = None
        chunk_strategy: Optional[str] = None
        # Per-stage timings (ms)
        parse_ms: Optional[int] = None
        chunk_ms: Optional[int] = None
        embed_ms: Optional[int] = None
        insert_ms: Optional[int] = None
        _maybe_print_counts(dbs, session_name, "baseline")
        for idx_f, filepath in enumerate(files, start=1):
            try:
                # Track per-file status row (create if missing; will update on completion)
                esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                if not esf:
                    esf = EmbeddingSessionFile(session_name=session_name, filepath=filepath, status="pending")
                    dbs.add(esf)
                    dbs.commit()
                elif getattr(esf, "status", None) == "complete":
                    # Skip already-completed files to support safe resume
                    if idx_f % 200 == 0:
                        print(f"[beta_worker] {session_name}: SKIP already complete {idx_f}/{total_files} -> {filepath}", flush=True)
                    continue

                current_stage = "parse"
                stage_start = time.time()
                with _deadline(PARSE_TIMEOUT):
                    base_doc = parse_file(filepath)
                current_text_len = len(base_doc.get("text", ""))
                parse_ms = int((time.time() - stage_start) * 1000)
                if not base_doc or not base_doc.get("text"):
                    esf.status = "error"
                    dbs.commit()
                    failures.append(filepath)
                    _append_log_line(log_dir, session_name, filepath, success=False)
                    print(f"[beta_worker] {session_name}: FAILED (empty) {idx_f}/{total_files} -> {filepath}", flush=True)
                    _append_error_detail(
                        log_dir, session_name, filepath,
                        current_stage or "parse", "EmptyText", "Parsed file has no text",
                        int((time.time() - stage_start) * 1000) if stage_start else None,
                        {"text_len": current_text_len}
                    )
                    continue

                # Enrich metadata
                path_meta = derive_path_metadata(filepath, root_dir or "")
                base_meta = dict(base_doc.get("chunk_metadata") or {})
                base_meta.update(path_meta)

                # Detect content type for analytics
                detected_type = detect_doc_type(base_meta, base_doc.get("text", ""))
                if detected_type and not base_meta.get("type"):
                    base_meta["type"] = detected_type

                # Semantic chunking
                current_stage = "chunk"
                stage_start = time.time()
                with _deadline(CHUNK_TIMEOUT):
                    # Try dashed-header aware chunking first (works for any dashed-header format).
                    file_chunks = chunk_legislation_dashed_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                    if file_chunks:
                        chunk_strategy = "dashed-semantic"
                    else:
                        # Fallback to generic semantic chunking (heading/sentences with same cfg)
                        file_chunks = chunk_document_semantic(base_doc["text"], base_meta=base_meta, cfg=cfg)
                        if file_chunks:
                            chunk_strategy = "semantic"
                    # Optional RCTS fallback for generic types (env-controlled)
                    if (not file_chunks) and os.environ.get("AUSLEGALSEARCH_USE_RCTS_GENERIC", "0") == "1":
                        rcts_chunks = chunk_generic_rcts(base_doc["text"], base_meta=base_meta, cfg=cfg)
                        if rcts_chunks:
                            file_chunks = rcts_chunks
                            chunk_strategy = "rcts-generic"
                current_chunk_count = len(file_chunks)
                chunk_ms = int((time.time() - stage_start) * 1000)
                if not file_chunks:
                    esf.status = "complete"
                    dbs.commit()
                    successes.append(filepath)
                    _append_success_metrics_line(
                        log_dir=log_dir,
                        session_name=session_name,
                        file_path=filepath,
                        chunks_count=0,
                        text_len=current_text_len or 0,
                        cfg=cfg,
                        strategy=chunk_strategy or "no-chunks",
                        detected_type=detected_type,
                        section_count=0,
                        tokens_est_total=0,
                        tokens_est_mean=0,
                    ,
                        parse_ms=parse_ms,
                        chunk_ms=chunk_ms,
                        embed_ms=None,
                        insert_ms=None,
                    )
                    print(f"[beta_worker] {session_name}: OK (0 chunks) {idx_f}/{total_files} -> {filepath}", flush=True)
                    continue

                # Batch embed
                texts = [c["text"] for c in file_chunks]
                current_stage = "embed"
                stage_start = time.time()
                vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                embed_ms = int((time.time() - stage_start) * 1000)

                # Insert
                current_stage = "insert"
                stage_start = time.time()
                inserted = _batch_insert_chunks(
                    session=dbs,
                    chunks=file_chunks,
                    vectors=vecs,
                    source_path=filepath,
                    fmt=base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
                )
                insert_ms = int((time.time() - stage_start) * 1000)
                processed_chunks += inserted

                # Update progress
                esf.status = "complete"
                dbs.commit()
                update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                successes.append(filepath)
                # Compute light-weight per-file metrics for logging (O(n) in chunks; negligible vs embedding)
                try:
                    section_idxs = set()
                    token_vals = []
                    for c in file_chunks:
                        md = c.get("chunk_metadata") or {}
                        if isinstance(md, dict):
                            si = md.get("section_idx")
                            if si is not None:
                                section_idxs.add(si)
                            tv = md.get("tokens_est")
                            if isinstance(tv, int):
                                token_vals.append(tv)
                    section_count = len(section_idxs) if section_idxs else 0
                    tokens_est_total = sum(token_vals) if token_vals else None
                    tokens_est_mean = int(round(tokens_est_total / len(token_vals))) if token_vals else None
                except Exception:
                    section_count = None
                    tokens_est_total = None
                    tokens_est_mean = None
                _append_success_metrics_line(
                    log_dir=log_dir,
                    session_name=session_name,
                    file_path=filepath,
                    chunks_count=current_chunk_count or 0,
                    text_len=current_text_len or 0,
                    cfg=cfg,
                    strategy=chunk_strategy or "semantic",
                    detected_type=detected_type,
                    section_count=section_count,
                    tokens_est_total=tokens_est_total,
                    tokens_est_mean=tokens_est_mean,
                    parse_ms=parse_ms,
                    chunk_ms=chunk_ms,
                    embed_ms=embed_ms,
                    insert_ms=insert_ms,
                )

                if idx_f % 10 == 0 or inserted > 0:
                    print(f"[beta_worker] {session_name}: OK {idx_f}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                if os.environ.get("AUSLEGALSEARCH_DEBUG_COUNTS", "0") == "1" and idx_f % 50 == 0:
                    _maybe_print_counts(dbs, session_name, f"after {idx_f} files")

            except _Timeout as e:
                # Try fallback if timeout occurred during chunking
                if (current_stage == "chunk"
                    and os.environ.get("AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT", "1") != "0"
                    and "base_doc" in locals()
                    and isinstance(base_doc, dict)
                    and base_doc.get("text")):
                    try:
                        fb_chunks = _fallback_chunk_text(base_doc["text"], base_meta, cfg)
                        texts = [c["text"] for c in fb_chunks]
                        t0 = time.time()
                        vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                        fb_embed_ms = int((time.time() - t0) * 1000)
                        t1 = time.time()
                        inserted = _batch_insert_chunks(
                            session=dbs,
                            chunks=fb_chunks,
                            vectors=vecs,
                            source_path=filepath,
                            fmt=base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
                        )
                        fb_insert_ms = int((time.time() - t1) * 1000)
                        processed_chunks += inserted
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                            if esf:
                                esf.status = "complete"
                                dbs.commit()
                        except Exception:
                            pass
                        update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                        successes.append(filepath)
                        _append_success_metrics_line(
                            log_dir=log_dir,
                            session_name=session_name,
                            file_path=filepath,
                            chunks_count=inserted,
                            text_len=len(base_doc.get("text","")),
                            cfg=cfg,
                            strategy="fallback-naive",
                            detected_type=detected_type,
                            parse_ms=parse_ms,
                            chunk_ms=None,
                            embed_ms=fb_embed_ms,
                            insert_ms=fb_insert_ms,
                        )
                        _append_error_detail(
                            log_dir, session_name, filepath,
                            "chunk", "Timeout", "Chunk timeout; fallback succeeded",
                            int((time.time() - stage_start) * 1000) if stage_start else None,
                            {"text_len": len(base_doc.get("text","")), "fallback": True, "fallback_chunks": len(fb_chunks), "batch_size": batch_size}
                        )
                        if idx_f % 10 == 0 or inserted > 0:
                            print(f"[beta_worker] {session_name}: OK (fallback) {idx_f}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                        continue
                    except Exception as e2:
                        # Fall through to treat as failure
                        e = e2
                # Failure path
                try:
                    esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                    if esf:
                        esf.status = "error"
                        dbs.commit()
                except Exception:
                    pass
                failures.append(filepath)
                _append_log_line(log_dir, session_name, filepath, success=False)
                print(f"[beta_worker] {session_name}: TIMEOUT {idx_f}/{total_files} -> {filepath} :: {e}", flush=True)
                _append_error_detail(
                    log_dir, session_name, filepath,
                    current_stage or "unknown", "Timeout", str(e),
                    int((time.time() - stage_start) * 1000) if stage_start else None,
                    {"text_len": current_text_len, "chunk_count": current_chunk_count, "batch_size": batch_size, "fallback": False}
                )
                continue
            except KeyboardInterrupt:
                print(f"[beta_worker] {session_name}: INTERRUPTED at {idx_f}/{total_files} -> {filepath}", flush=True)
                raise
            except Exception as e:
                # If failure in chunk stage, attempt naive fallback chunking
                if (current_stage == "chunk"
                    and os.environ.get("AUSLEGALSEARCH_FALLBACK_CHUNK_ON_TIMEOUT", "1") != "0"
                    and "base_doc" in locals()
                    and isinstance(base_doc, dict)
                    and base_doc.get("text")):
                    try:
                        fb_chunks = _fallback_chunk_text(base_doc["text"], base_meta, cfg)
                        texts = [c["text"] for c in fb_chunks]
                        t0 = time.time()
                        vecs = _embed_in_batches(embedder, texts, batch_size=batch_size)
                        fb_embed_ms = int((time.time() - t0) * 1000)
                        t1 = time.time()
                        inserted = _batch_insert_chunks(
                            session=dbs,
                            chunks=fb_chunks,
                            vectors=vecs,
                            source_path=filepath,
                            fmt=base_doc.get("format", Path(filepath).suffix.lower().strip(".")),
                        )
                        fb_insert_ms = int((time.time() - t1) * 1000)
                        processed_chunks += inserted
                        try:
                            esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                            if esf:
                                esf.status = "complete"
                                dbs.commit()
                        except Exception:
                            pass
                        update_session_progress(session_name, last_file=filepath, last_chunk=inserted - 1, processed_chunks=processed_chunks)
                        successes.append(filepath)
                        _append_success_metrics_line(
                            log_dir=log_dir,
                            session_name=session_name,
                            file_path=filepath,
                            chunks_count=inserted,
                            text_len=len(base_doc.get("text","")),
                            cfg=cfg,
                            strategy="fallback-naive",
                            detected_type=detected_type,
                            parse_ms=parse_ms,
                            chunk_ms=None,
                            embed_ms=fb_embed_ms,
                            insert_ms=fb_insert_ms,
                        )
                        _append_error_detail(
                            log_dir, session_name, filepath,
                            "chunk", type(e).__name__, "Primary chunker error; fallback succeeded",
                            int((time.time() - stage_start) * 1000) if stage_start else None,
                            {"text_len": len(base_doc.get("text","")), "fallback": True, "fallback_chunks": len(fb_chunks), "batch_size": batch_size},
                            traceback.format_exc()
                        )
                        if idx_f % 10 == 0 or inserted > 0:
                            print(f"[beta_worker] {session_name}: OK (fallback) {idx_f}/{total_files}, chunks+={inserted}, total_chunks={processed_chunks}", flush=True)
                        continue
                    except Exception as e2:
                        # Fall through to regular failure handling
                        e = e2
                # Regular failure handling
                try:
                    esf = dbs.query(EmbeddingSessionFile).filter_by(session_name=session_name, filepath=filepath).first()
                    if esf:
                        esf.status = "error"
                        dbs.commit()
                except Exception:
                    pass
                failures.append(filepath)
                _append_log_line(log_dir, session_name, filepath, success=False)
                print(f"[beta_worker] {session_name}: FAILED {idx_f}/{total_files} -> {filepath} :: {e}", flush=True)
                _append_error_detail(
                    log_dir, session_name, filepath,
                    current_stage or "unknown", type(e).__name__, str(e),
                    int((time.time() - stage_start) * 1000) if stage_start else None,
                    {"text_len": current_text_len, "chunk_count": current_chunk_count, "batch_size": batch_size},
                    traceback.format_exc()
                )
                continue

    # Write logs
    paths = _write_logs(log_dir, session_name, successes, failures)
    print(f"[beta_worker] Session {session_name} complete. Files OK: {len(successes)}, failed: {len(failures)}", flush=True)
    print(f"[beta_worker] Success log: {paths['success_log']}", flush=True)
    print(f"[beta_worker] Error log:   {paths['error_log']}", flush=True)

    # Mark session complete (do not hard-fail on partial errors; logs capture details)
    complete_session(session_name)


def _parse_cli_args(argv: List[str]) -> Dict[str, Any]:
    import argparse
    ap = argparse.ArgumentParser(description="Beta ingestion worker: semantic chunking + embeddings + DB write.")
    ap.add_argument("session_name", help="Child session name (e.g., sess-...-gpu0)")
    ap.add_argument("--root", default=None, help="Root directory (used if --partition_file not provided)")
    ap.add_argument("--partition_file", default=None, help="Text file listing file paths for this worker")
    ap.add_argument("--model", default=None, help="Embedding model (optional)")
    ap.add_argument("--target_tokens", type=int, default=512, help="Chunking target tokens (default 512)")
    ap.add_argument("--overlap_tokens", type=int, default=64, help="Chunk overlap tokens (default 64)")
    ap.add_argument("--max_tokens", type=int, default=640, help="Hard max per chunk (default 640)")
    ap.add_argument("--log_dir", default="./logs", help="Directory to write per-worker success/error logs")
    return vars(ap.parse_args(argv))


if __name__ == "__main__":
    args = _parse_cli_args(sys.argv[1:])
    run_worker(
        session_name=args["session_name"],
        root_dir=args.get("root"),
        partition_file=args.get("partition_file"),
        embedding_model=args.get("model"),
        token_target=args.get("target_tokens") or 512,
        token_overlap=args.get("overlap_tokens") or 64,
        token_max=args.get("max_tokens") or 640,
        log_dir=args.get("log_dir") or "./logs",
    )
