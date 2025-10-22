#!/usr/bin/env python3
"""
End-to-end SQL latency benchmark for AUSLegalSearch v3

Targets:
- Measure end-to-end latency for:
  - Vector similarity (pgvector) + JSON metadata filters
  - Full-text search (FTS) on documents.content (documents.document_fts)
  - JSON metadata-only filtering (no vector, no FTS)
  - Hybrid (client-side combine)
- Report p50/p95 across multiple runs to help tune for < 50ms p95

Supported metadata filters (aligned to stored/generated columns in schema-post-load/create_indexes.sql):
- type, jurisdiction, subjurisdiction, database
- date range (date_from..date_to), year
- title_eq (exact), author_eq (exact), citation (exact)
- title_member (titles[] membership), citation_member (citations[] membership)
- country (countries[] membership)
- source_approx (trigram on lower(documents.source))
- author (trigram on lower(metadata.author))
- title (trigram on lower(metadata.title))

Usage examples:
  python3 tools/bench_sql_latency.py \
    --query "Australia Peru investment agreement" \
    --top_k 10 \
    --runs 10 \
    --probes 10 \
    --hnsw_ef 60 \
    --type treaty \
    --subjurisdiction dfat \
    --jurisdiction au \
    --database ATS \
    --country "United States"

  python3 tools/bench_sql_latency.py \
    --query "Angelides v James Stedman Hendersons" \
    --top_k 10 \
    --runs 10 \
    --probes 12 \
    --type case \
    --jurisdiction cth \
    --database HCA \
    --title_member "Angelides v James Stedman Hendersons Sweets Ltd" \
    --citation_member "[1927] HCA 34"

Notes:
- Uses SQLAlchemy engine from db.connector (reads .env automatically).
- Embedding model comes from AUSLEGALSEARCH_EMBED_MODEL env or default in embedding/embedder.py.
- Tune ivfflat.probes (IVFFLAT) or hnsw.ef_search (HNSW) per run.
"""

import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Connection
from db.connector import engine
from embedding.embedder import Embedder


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _set_ivf_probes(conn: Connection, probes: Optional[int]) -> None:
    if probes and int(probes) > 0:
        conn.execute(text("SET LOCAL ivfflat.probes = :p"), {"p": int(probes)})


def _set_hnsw_ef(conn: Connection, ef_search: Optional[int]) -> None:
    if ef_search and int(ef_search) > 0:
        conn.execute(text("SET LOCAL hnsw.ef_search = :ef"), {"ef": int(ef_search)})


def _set_session_tuning(conn: Connection, use_jit: bool = False) -> None:
    # Disable JIT for short queries to reduce tail latency
    conn.execute(text("SET LOCAL jit = :jit"), {"jit": "on" if use_jit else "off"})


def _build_vector_array_sql(vec: List[float]) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {}
    parts = []
    for i, v in enumerate(vec):
        key = f"v{i}"
        parts.append(f":{key}")
        params[key] = float(v)
    array_sql = "ARRAY[" + ",".join(parts) + "]::vector"
    return array_sql, params


def run_vector_query(
    conn: Connection,
    query_vec: List[float],
    top_k: int,
    filters: Dict[str, Any],
    probes: Optional[int] = None,
    hnsw_ef: Optional[int] = None,
    use_jit: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)
    _set_ivf_probes(conn, probes)
    _set_hnsw_ef(conn, hnsw_ef)

    where_clauses = []
    params: Dict[str, Any] = {"topk": int(top_k)}

    # Equality filters on JSON metadata (prefer stored columns if present; fallback to JSONB)
    if filters.get("type"):
        where_clauses.append("(e.md_type = :type_exact OR (e.chunk_metadata->>'type') = :type_exact)")
        params["type_exact"] = str(filters["type"])

    if filters.get("jurisdiction"):
        where_clauses.append("(e.md_jurisdiction = :jurisdiction_exact OR (e.chunk_metadata->>'jurisdiction') = :jurisdiction_exact)")
        params["jurisdiction_exact"] = str(filters["jurisdiction"])

    if filters.get("subjurisdiction"):
        where_clauses.append("(e.md_subjurisdiction = :subjurisdiction_exact OR (e.chunk_metadata->>'subjurisdiction') = :subjurisdiction_exact)")
        params["subjurisdiction_exact"] = str(filters["subjurisdiction"])

    if filters.get("database"):
        where_clauses.append("(e.md_database = :database_exact OR (e.chunk_metadata->>'database') = :database_exact)")
        params["database_exact"] = str(filters["database"])

    if filters.get("year") is not None:
        where_clauses.append("(e.md_year = :year_exact OR ((e.chunk_metadata->>'year')::int) = :year_exact)")
        params["year_exact"] = int(filters["year"])

    if filters.get("date_from") and filters.get("date_to"):
        where_clauses.append("((e.md_date BETWEEN :df AND :dt) OR ((e.chunk_metadata->>'date')::date BETWEEN :df AND :dt))")
        params["df"] = str(filters["date_from"])
        params["dt"] = str(filters["date_to"])

    # Exact equality on title/author/citation
    if filters.get("title_eq"):
        where_clauses.append("(e.md_title = :title_eq OR (e.chunk_metadata->>'title') = :title_eq)")
        params["title_eq"] = str(filters["title_eq"])

    if filters.get("author_eq"):
        where_clauses.append("(e.md_author = :author_eq OR (e.chunk_metadata->>'author') = :author_eq)")
        params["author_eq"] = str(filters["author_eq"])

    if filters.get("citation"):
        where_clauses.append("(e.md_citation = :citation_eq OR (e.chunk_metadata->>'citation') = :citation_eq)")
        params["citation_eq"] = str(filters["citation"])

    # Membership tests on arrays
    if filters.get("country"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
              WHERE LOWER(c.val) = LOWER(:country)
            )
        """)
        params["country"] = str(filters["country"])

    if filters.get("title_member"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_titles, e.chunk_metadata->'titles')) AS t(val)
              WHERE LOWER(t.val) = LOWER(:title_member)
            )
        """)
        params["title_member"] = str(filters["title_member"])

    if filters.get("citation_member"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(COALESCE(e.md_citations, e.chunk_metadata->'citations')) AS c(val)
              WHERE LOWER(c.val) = LOWER(:citation_member)
            )
        """)
        params["citation_member"] = str(filters["citation_member"])

    # Approximate matches (trigram); rely on md_*_lc or JSONB fallback
    if filters.get("author"):
        where_clauses.append("(e.md_author_lc % LOWER(:author_approx) OR LOWER(coalesce(e.chunk_metadata->>'author','')) % LOWER(:author_approx))")
        params["author_approx"] = str(filters["author"])

    if filters.get("title"):
        where_clauses.append("(e.md_title_lc % LOWER(:title_approx) OR LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx))")
        params["title_approx"] = str(filters["title"])

    if filters.get("source_approx"):
        where_clauses.append("""
            EXISTS (
              SELECT 1
              FROM documents d
              WHERE d.id = e.doc_id
                AND (d.source_lc % LOWER(:src_approx) OR LOWER(d.source) % LOWER(:src_approx))
            )
        """)
        params["src_approx"] = str(filters["source_approx"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Vector array
    array_sql, arr_params = _build_vector_array_sql(query_vec)
    params.update(arr_params)

    sql = f"""
    SELECT e.doc_id, e.chunk_index,
           (e.vector <#> {array_sql}) AS distance,
           d.source,
           e.chunk_metadata
    FROM embeddings e
    JOIN documents d ON d.id = e.doc_id
    {where_sql}
    ORDER BY e.vector <#> {array_sql} ASC
    LIMIT :topk
    """
    rows = conn.execute(text(sql), params).fetchall()
    hits = []
    for r in rows:
        hits.append({
            "doc_id": r[0],
            "chunk_index": r[1],
            "distance": float(r[2]),
            "source": r[3],
            "chunk_metadata": r[4],
        })
    t1 = _now_ms()
    return hits, (t1 - t0)


def run_fts_query(
    conn: Connection,
    query: str,
    top_k: int,
    use_jit: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)
    sql = """
    WITH q AS (SELECT plainto_tsquery('english', :q) AS ts)
    SELECT d.id, d.source,
           ts_rank(d.document_fts, (SELECT ts FROM q)) AS rank
    FROM documents d, q
    WHERE d.document_fts @@ (SELECT ts FROM q)
    ORDER BY rank DESC
    LIMIT :k
    """
    rows = conn.execute(text(sql), {"q": query, "k": int(top_k)}).fetchall()
    hits = [{"doc_id": r[0], "source": r[1], "rank": float(r[2])} for r in rows]
    t1 = _now_ms()
    return hits, (t1 - t0)


def hybrid_rerank(vec_hits: List[Dict[str, Any]], fts_hits: List[Dict[str, Any]], alpha: float = 0.5, top_k: int = 10) -> List[Dict[str, Any]]:
    # Map FTS ranks to a normalized score (simple min-max)
    f_map = {}
    if fts_hits:
        ranks = [h["rank"] for h in fts_hits]
        rmin, rmax = min(ranks), max(ranks)
        for h in fts_hits:
            f_map[(h["doc_id"])] = 1.0 if rmin == rmax else (h["rank"] - rmin) / (rmax - rmin)

    # Normalize vector distances to similarity
    v_map = {}
    if vec_hits:
        dists = [h["distance"] for h in vec_hits]
        dmin, dmax = min(dists), max(dists)
        for h in vec_hits:
            sim = 1.0 if dmax == dmin else 1.0 - ((h["distance"] - dmin) / (dmax - dmin))
            v_map[(h["doc_id"], h.get("chunk_index", 0))] = sim

    # Combine by doc_id; keep the best chunk per doc for display
    combined: Dict[int, Dict[str, Any]] = {}
    for vh in vec_hits:
        key = vh["doc_id"]
        vs = v_map.get((vh["doc_id"], vh.get("chunk_index", 0)), 0.0)
        fs = f_map.get(key, 0.0)
        score = alpha * vs + (1 - alpha) * fs
        prev = combined.get(key)
        if (prev is None) or (score > prev["score"]):
            combined[key] = {"doc_id": key, "source": vh["source"], "score": score, "chunk_index": vh.get("chunk_index", 0)}

    for fh in fts_hits:
        key = fh["doc_id"]
        vs = 0.0  # if not in vec map
        fs = f_map.get(key, 0.0)
        score = alpha * vs + (1 - alpha) * fs
        prev = combined.get(key)
        if (prev is None) or (score > prev["score"]):
            combined[key] = {"doc_id": key, "source": fh["source"], "score": score}

    out = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return out


def run_metadata_filter_query(
    conn: Connection,
    top_k: int,
    filters: Dict[str, Any],
    use_jit: bool = False
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Measure latency for JSONB metadata filtering only (no vector, no FTS).
    Useful to validate btree/GIN indexes and selectivity.
    """
    t0 = _now_ms()
    _set_session_tuning(conn, use_jit=use_jit)

    where_clauses = []
    params: Dict[str, Any] = {"topk": int(top_k)}

    # Equality filters on stored columns or JSONB fallback
    if filters.get("type"):
        where_clauses.append("(e.md_type = :type_exact OR (e.chunk_metadata->>'type') = :type_exact)")
        params["type_exact"] = str(filters["type"])

    if filters.get("jurisdiction"):
        where_clauses.append("(e.md_jurisdiction = :jurisdiction_exact OR (e.chunk_metadata->>'jurisdiction') = :jurisdiction_exact)")
        params["jurisdiction_exact"] = str(filters["jurisdiction"])

    if filters.get("subjurisdiction"):
        where_clauses.append("(e.md_subjurisdiction = :subjurisdiction_exact OR (e.chunk_metadata->>'subjurisdiction') = :subjurisdiction_exact)")
        params["subjurisdiction_exact"] = str(filters["subjurisdiction"])

    if filters.get("database"):
        where_clauses.append("(e.md_database = :database_exact OR (e.chunk_metadata->>'database') = :database_exact)")
        params["database_exact"] = str(filters["database"])

    if filters.get("year") is not None:
        where_clauses.append("(e.md_year = :year_exact OR ((e.chunk_metadata->>'year')::int) = :year_exact)")
        params["year_exact"] = int(filters["year"])

    if filters.get("date_from") and filters.get("date_to"):
        where_clauses.append("((e.md_date BETWEEN :df AND :dt) OR ((e.chunk_metadata->>'date')::date BETWEEN :df AND :dt))")
        params["df"] = str(filters["date_from"])
        params["dt"] = str(filters["date_to"])

    # Exact equality on title/author/citation
    if filters.get("title_eq"):
        where_clauses.append("(e.md_title = :title_eq OR (e.chunk_metadata->>'title') = :title_eq)")
        params["title_eq"] = str(filters["title_eq"])

    if filters.get("author_eq"):
        where_clauses.append("(e.md_author = :author_eq OR (e.chunk_metadata->>'author') = :author_eq)")
        params["author_eq"] = str(filters["author_eq"])

    if filters.get("citation"):
        where_clauses.append("(e.md_citation = :citation_eq OR (e.chunk_metadata->>'citation') = :citation_eq)")
        params["citation_eq"] = str(filters["citation"])

    # Membership tests on arrays
    if filters.get("country"):
        where_clauses.append("""
          EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_countries, e.chunk_metadata->'countries')) AS c(val)
            WHERE LOWER(c.val) = LOWER(:country)
          )
        """)
        params["country"] = str(filters["country"])

    if filters.get("title_member"):
        where_clauses.append("""
          EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_titles, e.chunk_metadata->'titles')) AS t(val)
            WHERE LOWER(t.val) = LOWER(:title_member)
          )
        """)
        params["title_member"] = str(filters["title_member"])

    if filters.get("citation_member"):
        where_clauses.append("""
          EXISTS (
            SELECT 1
            FROM jsonb_array_elements_text(COALESCE(e.md_citations, e.chunk_metadata->'citations')) AS c(val)
            WHERE LOWER(c.val) = LOWER(:citation_member)
          )
        """)
        params["citation_member"] = str(filters["citation_member"])

    # Approximate matches (trigram)
    if filters.get("author"):
        where_clauses.append("(e.md_author_lc % LOWER(:author_approx) OR LOWER(coalesce(e.chunk_metadata->>'author','')) % LOWER(:author_approx))")
        params["author_approx"] = str(filters["author"])

    if filters.get("title"):
        where_clauses.append("(e.md_title_lc % LOWER(:title_approx) OR LOWER(coalesce(e.chunk_metadata->>'title','')) % LOWER(:title_approx))")
        params["title_approx"] = str(filters["title"])

    if filters.get("source_approx"):
        where_clauses.append("""
          EXISTS (
            SELECT 1 FROM documents d
            WHERE d.id = e.doc_id
              AND (d.source_lc % LOWER(:src_approx) OR LOWER(d.source) % LOWER(:src_approx))
          )
        """)
        params["src_approx"] = str(filters["source_approx"])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
    SELECT e.doc_id, e.chunk_index, e.chunk_metadata
    FROM embeddings e
    {where_sql}
    LIMIT :topk
    """
    rows = conn.execute(text(sql), params).fetchall()
    hits = [{"doc_id": r[0], "chunk_index": r[1], "chunk_metadata": r[2]} for r in rows]
    t1 = _now_ms()
    return hits, (t1 - t0)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s)-1, int(round((p/100.0) * (len(s)-1)))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description="Benchmark SQL latency for vector/metadata/FTS searches.")
    ap.add_argument("--query", required=True, help="Search query text")
    ap.add_argument("--top_k", type=int, default=10, help="Top K results")
    ap.add_argument("--probes", type=int, default=10, help="ivfflat.probes for pgvector")
    ap.add_argument("--hnsw_ef", type=int, default=0, help="hnsw.ef_search for HNSW (pgvector >= 0.7). 0 to skip")
    ap.add_argument("--runs", type=int, default=5, help="Number of repetitions per test to compute p50/p95")
    ap.add_argument("--use_jit", action="store_true", help="Enable JIT (off by default for lower tail)")

    # Equality filters
    ap.add_argument("--type", dest="type_", default=None, help="Exact metadata type (case, legislation, journal, treaty, etc.)")
    ap.add_argument("--jurisdiction", default=None, help="Exact metadata jurisdiction (e.g., au, cth)")
    ap.add_argument("--subjurisdiction", default=None, help="Exact metadata subjurisdiction (e.g., nsw, dfat)")
    ap.add_argument("--database", default=None, help="Exact metadata database (e.g., HCA, consol_act, UNSWLawJl, ATS)")
    ap.add_argument("--year", type=int, default=None, help="Exact metadata year")
    ap.add_argument("--date_from", default=None, help="YYYY-MM-DD")
    ap.add_argument("--date_to", default=None, help="YYYY-MM-DD")
    ap.add_argument("--title_eq", default=None, help="Exact title")
    ap.add_argument("--author_eq", default=None, help="Exact author")
    ap.add_argument("--citation", default=None, help="Exact single 'citation' field (journals)")

    # Membership filters
    ap.add_argument("--title_member", default=None, help="Check membership in titles[]")
    ap.add_argument("--citation_member", default=None, help="Check membership in citations[]")
    ap.add_argument("--country", default=None, help="Country membership (countries[])")

    # Approximate (trigram) filters
    ap.add_argument("--source_approx", default=None, help="Approximate match for documents.source")
    ap.add_argument("--author", default=None, help="Approximate author name")
    ap.add_argument("--title", default=None, help="Approximate title")

    args = ap.parse_args()

    embedder = Embedder()
    query_vec = embedder.embed([args.query])[0].tolist()

    filters = {
        "type": args.type_,
        "jurisdiction": args.jurisdiction,
        "subjurisdiction": args.subjurisdiction,
        "database": args.database,
        "year": args.year,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "title_eq": args.title_eq,
        "author_eq": args.author_eq,
        "citation": args.citation,
        "title_member": args.title_member,
        "citation_member": args.citation_member,
        "country": args.country,
        "source_approx": args.source_approx,
        "author": args.author,
        "title": args.title,
    }

    with engine.begin() as conn:
        # Warmup and session tune
        _set_session_tuning(conn, use_jit=args.use_jit)
        _set_ivf_probes(conn, args.probes)
        _set_hnsw_ef(conn, args.hnsw_ef)
        conn.execute(text("SELECT 1"))

        # Repeated runs to compute p50/p95
        vec_times: List[float] = []
        fts_times: List[float] = []
        md_times: List[float] = []
        last_vec_hits: List[Dict[str, Any]] = []
        last_fts_hits: List[Dict[str, Any]] = []
        last_md_hits: List[Dict[str, Any]] = []

        for _ in range(max(1, args.runs)):
            vec_hits, vec_ms = run_vector_query(conn, query_vec, args.top_k, filters, probes=args.probes, hnsw_ef=args.hnsw_ef, use_jit=args.use_jit)
            vec_times.append(vec_ms)
            last_vec_hits = vec_hits

        for _ in range(max(1, args.runs)):
            fts_hits, fts_ms = run_fts_query(conn, args.query, args.top_k, use_jit=args.use_jit)
            fts_times.append(fts_ms)
            last_fts_hits = fts_hits

        for _ in range(max(1, args.runs)):
            md_hits, md_ms = run_metadata_filter_query(conn, args.top_k, filters, use_jit=args.use_jit)
            md_times.append(md_ms)
            last_md_hits = md_hits

        # Hybrid combine (client-side rerank) on the last observed vec/fts
        t0 = _now_ms()
        hybrid = hybrid_rerank(last_vec_hits, last_fts_hits, alpha=0.5, top_k=args.top_k)
        hybrid_ms = _now_ms() - t0

        def _summary_line(name: str, values: List[float]) -> str:
            if not values:
                return f"{name}: n=0"
            return f"{name}: p50={_percentile(values,50):.2f} ms  p95={_percentile(values,95):.2f} ms  n={len(values)}"

        print("\n--- Latency Summary (ms) ---")
        print(_summary_line("Vector (pgvector+filters)", vec_times))
        print(_summary_line("FTS (documents_fts)", fts_times))
        print(_summary_line("Metadata-only JSON filters", md_times))
        print(f"Hybrid combine (client): {hybrid_ms:.2f} ms")

        print("\n--- Top Vector Hits ---")
        for h in last_vec_hits[:args.top_k]:
            cm = h.get("chunk_metadata") or {}
            print(f"doc_id={h['doc_id']} chunk={h['chunk_index']} dist={h['distance']:.4f} src={h['source'][:90]} meta_title={cm.get('title')}")

        print("\n--- Top FTS Hits ---")
        for h in last_fts_hits[:args.top_k]:
            print(f"doc_id={h['doc_id']} rank={h['rank']:.4f} src={h['source'][:120]}")

        print("\n--- Top Metadata-only Hits ---")
        for h in last_md_hits[:args.top_k]:
            cm = h.get("chunk_metadata") or {}
            print(f"doc_id={h['doc_id']} chunk={h['chunk_index']} title={cm.get('title')} type={cm.get('type')} database={cm.get('database')}")

        print("\n--- Top Hybrid (by doc) ---")
        for h in hybrid[:args.top_k]:
            print(f"doc_id={h['doc_id']} score={h['score']:.4f} src={h['source'][:120]}")


if __name__ == "__main__":
    main()
