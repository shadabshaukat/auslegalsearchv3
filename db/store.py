"""
Centralized DB models and ORM for auslegalsearchv3.
- Exports all tables, full CRUD/session logic for users, ingestion, search, embedding, chat, and conversion files.
- All app code imports models and functions from here, schema created with create_all_tables().
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, Float, Date
from sqlalchemy import select, desc, text
from db.connector import engine, SessionLocal, Vector, JSONB, UUIDType
from datetime import datetime
import uuid
import os
import bcrypt

# Production: avoid loading ML models at import-time in DB module.
# Use a configured embedding dimension (defaults to 768 for common ST/HF models).
EMBEDDING_DIM = int(os.environ.get("AUSLEGALSEARCH_EMBED_DIM", "768"))

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=True)
    registered_google = Column(Boolean, default=False)
    google_id = Column(String, nullable=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    format = Column(String, nullable=False)

class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, ForeignKey('documents.id'), index=True)
    chunk_index = Column(Integer, nullable=False)
    vector = Column(Vector(EMBEDDING_DIM), nullable=False)
    chunk_metadata = Column(JSONB, nullable=True)
    document = relationship("Document", backref="embeddings")

class EmbeddingSession(Base):
    __tablename__ = "embedding_sessions"
    id = Column(Integer, primary_key=True)
    session_name = Column(String, unique=True, nullable=False)
    directory = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="active")
    last_file = Column(String, nullable=True)
    last_chunk = Column(Integer, nullable=True)
    total_files = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=True)
    processed_chunks = Column(Integer, nullable=True)

class EmbeddingSessionFile(Base):
    __tablename__ = "embedding_session_files"
    id = Column(Integer, primary_key=True)
    session_name = Column(String, nullable=False, index=True)
    filepath = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    completed_at = Column(DateTime, nullable=True)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(UUIDType(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    username = Column(String, nullable=True)
    question = Column(Text, nullable=True)
    chat_history = Column(JSONB, nullable=False)
    llm_params = Column(JSONB, nullable=False)

class ConversionFile(Base):
    __tablename__ = "conversion_files"
    id = Column(Integer, primary_key=True)
    session_name = Column(String, nullable=False, index=True)
    src_file = Column(String, nullable=False)
    dst_file = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    success = Column(Boolean, nullable=True, default=None)
    error_message = Column(Text, nullable=True)

# --- Relational tables for normalized legal metadata ---

class Case(Base):
    __tablename__ = "cases"
    case_id = Column(Integer, primary_key=True)
    url = Column(String(1024), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    case_date = Column(Date, nullable=True)
    court = Column(String(128), nullable=True)

class CaseName(Base):
    __tablename__ = "case_names"
    case_name_id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False, index=True)
    name = Column(Text, nullable=False)

class CaseCitationRef(Base):
    __tablename__ = "case_citation_refs"
    citation_ref_id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

class Legislation(Base):
    __tablename__ = "legislation"
    legislation_id = Column(Integer, primary_key=True)
    url = Column(String(1024), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    enacted_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    name = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class LegislationSection(Base):
    __tablename__ = "legislation_sections"
    section_id = Column(Integer, primary_key=True)
    legislation_id = Column(Integer, ForeignKey('legislation.legislation_id'), nullable=False, index=True)
    identifier = Column(String(32), nullable=True)  # e.g., "288", "1.5.1"
    type = Column(String(32), nullable=True)        # e.g., "regulation", "schedule", "section"
    title = Column(Text, nullable=True)
    content = Column(Text, nullable=False)

# --- Journals (normalized) ---

class Journal(Base):
    __tablename__ = "journals"
    journal_id = Column(Integer, primary_key=True)
    url = Column(String(1024), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    published_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    title = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class JournalAuthor(Base):
    __tablename__ = "journal_authors"
    journal_author_id = Column(Integer, primary_key=True)
    journal_id = Column(Integer, ForeignKey('journals.journal_id'), nullable=False, index=True)
    name = Column(Text, nullable=False)

class JournalCitationRef(Base):
    __tablename__ = "journal_citation_refs"
    citation_ref_id = Column(Integer, primary_key=True)
    journal_id = Column(Integer, ForeignKey('journals.journal_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

# --- Treaties (normalized) ---

class Treaty(Base):
    __tablename__ = "treaties"
    treaty_id = Column(Integer, primary_key=True)
    url = Column(String(1024), nullable=True)
    jurisdiction = Column(String(32), nullable=True)
    subjurisdiction = Column(String(32), nullable=True)
    signed_date = Column(Date, nullable=True)
    year = Column(Integer, nullable=True)
    title = Column(Text, nullable=True)
    database = Column(String(128), nullable=True)

class TreatyCountry(Base):
    __tablename__ = "treaty_countries"
    treaty_country_id = Column(Integer, primary_key=True)
    treaty_id = Column(Integer, ForeignKey('treaties.treaty_id'), nullable=False, index=True)
    country = Column(String(128), nullable=False)

class TreatyCitationRef(Base):
    __tablename__ = "treaty_citation_refs"
    citation_ref_id = Column(Integer, primary_key=True)
    treaty_id = Column(Integer, ForeignKey('treaties.treaty_id'), nullable=False, index=True)
    citation = Column(Text, nullable=False)

def create_all_tables():
    # Enable vital extensions (in superuser context if allowed)
    exts = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS pg_trgm",
        "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"",
        "CREATE EXTENSION IF NOT EXISTS fuzzystrmatch",
    ]
    with engine.begin() as conn:
        for ddl in exts:
            try:
                conn.execute(text(ddl))
            except Exception as e:
                print(f"[EXT NOTE] Could not enable extension: {ddl}\nReason: {e}")

    Base.metadata.create_all(engine, tables=[
        User.__table__, Document.__table__, Embedding.__table__,
        EmbeddingSession.__table__, EmbeddingSessionFile.__table__,
        ChatSession.__table__, ConversionFile.__table__
    ])
    # --- Post-table DDL: create indexes, triggers, and FTS structures if missing ---
    # Light-init mode skips heavy backfills and large index builds to avoid stalls on new instances.
    # Set AUSLEGALSEARCH_SCHEMA_LIGHT_INIT=1 in the environment to enable.
    LIGHT_INIT = os.environ.get("AUSLEGALSEARCH_SCHEMA_LIGHT_INIT", "0") == "1"

    ddl_sql = [
        # 1. Add document_fts if missing
        """
        ALTER TABLE public.documents
        ADD COLUMN IF NOT EXISTS document_fts tsvector
        """,
    ]

    # 2. Populate document_fts (heavy backfill) — skip in LIGHT_INIT
    if not LIGHT_INIT:
        ddl_sql.append(
            """
            UPDATE public.documents
            SET document_fts = to_tsvector('english', coalesce(content, ''))
            WHERE document_fts IS NULL
            """
        )

    # 3-6. Indexes and trigger for FTS maintenance
    ddl_sql.extend([
        # 3. GIN FTS index for document_fts
        """
        CREATE INDEX IF NOT EXISTS idx_documents_fts
        ON public.documents USING GIN (document_fts)
        """,
        # 4. Trigram content index
        """
        CREATE INDEX IF NOT EXISTS idx_documents_content_trgm
        ON public.documents USING GIN (content gin_trgm_ops)
        """,
        # 5. Trigger function for updating document_fts
        """
        CREATE OR REPLACE FUNCTION documents_fts_trigger() RETURNS trigger AS $$
        BEGIN
          NEW.document_fts := to_tsvector('english', coalesce(NEW.content, ''));
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
        """,
        # 6. Drop old trigger if exists for safety, then create
        """
        DROP TRIGGER IF EXISTS tsvectorupdate ON public.documents
        """,
        """
        CREATE TRIGGER tsvectorupdate
        BEFORE INSERT OR UPDATE ON public.documents
        FOR EACH ROW EXECUTE FUNCTION documents_fts_trigger()
        """,
    ])

    # 7. IVFFLAT vector index for cosine on embeddings — skip build in LIGHT_INIT
    if not LIGHT_INIT:
        ddl_sql.append(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_vector_ivfflat_cosine
            ON public.embeddings USING ivfflat (vector vector_cosine_ops)
            WITH (lists = 100)
            """
        )

    # Ensure we never create duplicate session-file rows on retries/resumes
    ddl_sql.extend([
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_esf_session_file
        ON public.embedding_session_files (session_name, filepath)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_esf_status
        ON public.embedding_session_files (status)
        """,
    ])
    # Force psql to continue on error for objects already existing
    with engine.begin() as conn:
        for ddl in ddl_sql:
            try:
                conn.execute(text(ddl))
            except Exception as e:
                print(f"[DDL NOTE] Could not execute:\n{ddl}\nReason: {e}")


# --- User CRUD and Auth logic ---
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashval: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashval.encode('utf-8'))

def create_user(email, password=None, name=None, google_id=None, registered_google=False):
    with SessionLocal() as session:
        user = User(
            email=email,
            password_hash=hash_password(password) if password else None,
            name=name,
            google_id=google_id,
            registered_google=registered_google,
            created_at=datetime.utcnow(),
            last_login=datetime.utcnow(),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

def get_user_by_email(email: str):
    with SessionLocal() as session:
        return session.query(User).filter_by(email=email).first()

def set_last_login(user_id: int):
    with SessionLocal() as session:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            user.last_login = datetime.utcnow()
            session.commit()

def get_user_by_googleid(google_id: str):
    with SessionLocal() as session:
        return session.query(User).filter_by(google_id=google_id).first()

# -- Chat Session functions --
def save_chat_session(chat_history, llm_params, ended_at=None, username=None, question=None):
    with SessionLocal() as session:
        chat_sess = ChatSession(
            chat_history=chat_history,
            llm_params=llm_params,
            ended_at=ended_at or datetime.utcnow(),
            username=username,
            question=question
        )
        session.add(chat_sess)
        session.commit()
        session.refresh(chat_sess)
        return chat_sess.id

def get_chat_session(chat_id):
    with SessionLocal() as session:
        return session.query(ChatSession).filter_by(id=chat_id).first()

# ---- Embedding/DOC ingest/session tracking ----
def start_session(session_name, directory, total_files=None, total_chunks=None):
    with SessionLocal() as session:
        sess = EmbeddingSession(
            session_name=session_name,
            directory=directory,
            started_at=datetime.utcnow(),
            status="active",
            total_files=total_files,
            total_chunks=total_chunks,
            processed_chunks=0
        )
        session.add(sess)
        session.commit()
        session.refresh(sess)
        return sess

def update_session_progress(session_name, last_file, last_chunk, processed_chunks):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.last_file = last_file
        sess.last_chunk = last_chunk
        sess.processed_chunks = processed_chunks
        session.commit()
        return sess

def complete_session(session_name):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.ended_at = datetime.utcnow()
        sess.status = "complete"
        session.commit()
        return sess

def fail_session(session_name):
    with SessionLocal() as session:
        sess = session.query(EmbeddingSession).filter_by(session_name=session_name).first()
        if not sess:
            return None
        sess.status = "error"
        session.commit()
        return sess

def get_active_sessions():
    with SessionLocal() as session:
        sessions = session.query(EmbeddingSession).filter(EmbeddingSession.status == "active").all()
        return sessions

def get_resume_sessions():
    with SessionLocal() as session:
        return session.query(EmbeddingSession).filter(EmbeddingSession.status != "complete").all()

def get_session(session_name):
    with SessionLocal() as session:
        return session.query(EmbeddingSession).filter_by(session_name=session_name).first()

def add_document(doc: dict) -> int:
    with SessionLocal() as session:
        doc_obj = Document(
            source=doc["source"],
            content=doc["content"],
            format=doc["format"],
        )
        session.add(doc_obj)
        session.commit()
        session.refresh(doc_obj)
        return doc_obj.id

def add_embedding(doc_id: int, chunk_index: int, vector, chunk_metadata=None) -> int:
    with SessionLocal() as session:
        embed_obj = Embedding(
            doc_id=doc_id,
            chunk_index=chunk_index,
            vector=vector,
            chunk_metadata=chunk_metadata,
        )
        session.add(embed_obj)
        session.commit()
        session.refresh(embed_obj)
        return embed_obj.id

def search_vector(query_vec, top_k=5):
    with SessionLocal() as session:
        ndim = len(query_vec)
        array_params = []
        param_dict = {}
        for i, v in enumerate(query_vec):
            pname = f"v{i}"
            array_params.append(f":{pname}")
            param_dict[pname] = float(v)
        param_dict['topk'] = top_k
        array_sql = "ARRAY[" + ",".join(array_params) + "]::vector"
        sql = f'''
        SELECT embeddings.doc_id, embeddings.chunk_index, embeddings.vector <=> {array_sql} AS score,
            documents.content, documents.source, documents.format, embeddings.chunk_metadata
        FROM embeddings
        JOIN documents ON embeddings.doc_id = documents.id
        ORDER BY embeddings.vector <=> {array_sql} ASC
        LIMIT :topk
        '''
        result = session.execute(text(sql), param_dict)
        hits = []
        for row in result:
            hits.append({
                "doc_id": row[0],
                "chunk_index": row[1],
                "score": row[2],
                "text": row[3],
                "source": row[4],
                "format": row[5],
                "chunk_metadata": row[6],
            })
        return hits

def search_bm25(query, top_k=5):
    with SessionLocal() as session:
        q = f"%{query}%"
        res = session.execute(
            text("""
                SELECT id, content, source, format, NULL as chunk_metadata
                FROM documents
                WHERE content ILIKE :q
                LIMIT :topk
            """), {"q": q, "topk": top_k}
        )
        hits = []
        for row in res:
            hits.append({
                "doc_id": row[0],
                "chunk_index": 0,
                "score": 1.0,
                "text": row[1],
                "source": row[2],
                "format": row[3],
                "chunk_metadata": row[4],
            })
        return hits

def search_hybrid(query, top_k=5, alpha=0.5):
    # Lazy import to avoid heavy model load at module import-time
    from embedding.embedder import Embedder
    embedder = Embedder()
    query_vec = embedder.embed([query])[0]
    vector_hits = search_vector(query_vec, top_k=top_k * 2)
    bm25_hits = search_bm25(query, top_k=top_k * 2)

    all_hits = {}
    for h in vector_hits:
        key = (h["doc_id"], h["chunk_index"])
        all_hits[key] = {
            **h,
            "vector_score": h["score"],
            "bm25_score": 0.0,
            "hybrid_score": 0.0,
        }
    for h in bm25_hits:
        key = (h["doc_id"], h["chunk_index"])
        if key in all_hits:
            all_hits[key]["bm25_score"] = 1.0
        else:
            all_hits[key] = {
                **h,
                "vector_score": 0.0,
                "bm25_score": 1.0,
                "hybrid_score": 0.0,
            }
    scores = [v["vector_score"] for v in all_hits.values()]
    if scores:
        minv, maxv = min(scores), max(scores)
        for v in all_hits.values():
            if maxv != minv:
                v["vector_score_norm"] = 1.0 - ((v["vector_score"] - minv) / (maxv - minv))
            else:
                v["vector_score_norm"] = 1.0
    else:
        for v in all_hits.values():
            v["vector_score_norm"] = 0.0
    for v in all_hits.values():
        v["hybrid_score"] = alpha * v["vector_score_norm"] + (1 - alpha) * v["bm25_score"]
    results = sorted(all_hits.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
    for r in results:
        r["citation"] = f'{r["source"]}#chunk{r.get("chunk_index",0)}'
    return results

def get_file_contents(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        try:
            with open(filepath, "r", encoding="latin-1") as f:
                return f.read()
        except Exception:
            return f"Could not read file: {filepath}"

def add_conversion_file(session_name, src_file, dst_file, status="pending"):
    with SessionLocal() as session:
        cf = ConversionFile(
            session_name=session_name,
            src_file=src_file,
            dst_file=dst_file,
            status=status,
            start_time=datetime.utcnow() if status == "pending" else None,
            success=None
        )
        session.add(cf)
        session.commit()
        return cf.id

def update_conversion_file_status(cf_id, status, error_message=None, success=None):
    with SessionLocal() as session:
        cf = session.query(ConversionFile).filter_by(id=cf_id).first()
        cf.status = status
        cf.end_time = datetime.utcnow()
        if error_message:
            cf.error_message = error_message
        if success is not None:
            cf.success = success
        session.commit()
        return cf

def search_fts(query, top_k=10, mode="both"):
    """
    Full text search over 'document_fts' of documents and/or embeddings.chunk_metadata (jsonb as text).
    mode: "documents", "metadata", or "both"
    Deduplicates by metadata URL (if present), else doc_id/source, to return unique cases only.
    """
    import json as _jsonmod
    def extract_url(md):
        """Extracts URL from stringified or dict metadata, or returns None."""
        if not md:
            return None
        if isinstance(md, dict):
            return md.get("url")
        try:
            mdict = _jsonmod.loads(md) if isinstance(md, str) else md
            return mdict.get("url")
        except Exception:
            return None

    with SessionLocal() as session:
        all_hits = []
        seen_keys = set()

        if mode in ("documents", "both"):
            doc_sql = """
            SELECT id as doc_id, NULL as chunk_index,
                source, content, NULL as chunk_text,
                NULL as chunk_metadata,
                ts_headline('english', content, q, 'StartSel=<b>, StopSel=</b>') as snippet
            FROM documents, plainto_tsquery('english', :q) q
            WHERE document_fts @@ q
            LIMIT :topk
            """
            doc_hits = session.execute(text(doc_sql), {"q": query, "topk": top_k*8}).fetchall()
            for row in doc_hits:
                hit = {
                    "doc_id": row[0],
                    "chunk_index": row[1],
                    "source": row[2],
                    "content": row[3],
                    "text": row[4] if row[4] else row[3],
                    "chunk_metadata": row[5],
                    "snippet": row[6],
                    "search_area": "documents"
                }
                url = extract_url(row[5])
                hit["dedup_key"] = url if url else ("doc", row[0])
                all_hits.append(hit)

        if mode in ("metadata", "both"):
            chunk_sql = """
            SELECT e.doc_id, e.chunk_index, d.source, d.content,
                NULL as snippet,
                e.chunk_metadata::text as chunk_metadata,
                ts_headline('english', e.chunk_metadata::text, q, 'StartSel=<b>, StopSel=</b>') as chunk_snippet
            FROM embeddings e
            JOIN documents d ON e.doc_id = d.id,
                 plainto_tsquery('english', :q) q
            WHERE to_tsvector('english', e.chunk_metadata::text) @@ q
            LIMIT :topk
            """
            chunk_hits = session.execute(text(chunk_sql), {"q": query, "topk": top_k*8}).fetchall()
            for row in chunk_hits:
                hit = {
                    "doc_id": row[0],
                    "chunk_index": row[1],
                    "source": row[2],
                    "content": row[3],
                    "text": row[5],
                    "chunk_metadata": row[5],
                    "snippet": row[6],
                    "search_area": "metadata"
                }
                url = extract_url(row[5])
                hit["dedup_key"] = url if url else ("doc", row[0])
                all_hits.append(hit)

        # Group by dedup_key (url or doc_id) and select the "best" (lowest chunk_index/first) per group
        grouped = {}
        for h in all_hits:
            key = h.get("dedup_key")
            if key not in grouped:
                grouped[key] = h
            else:
                # If metadata, prefer chunk_index==0; else keep first seen
                if h["search_area"] == "metadata" and grouped[key]["search_area"] == "metadata":
                    if h.get("chunk_index", 9999) < grouped[key].get("chunk_index", 9999):
                        grouped[key] = h

        unique_hits = list(grouped.values())
        # Optionally, rank: for now keep order, but slice to top_k
        return unique_hits[:top_k]


__all__ = [
    "Base", "engine", "SessionLocal", "Vector", "JSONB", "UUIDType",
    "User", "Document", "Embedding", "EmbeddingSession", "EmbeddingSessionFile", "ChatSession", "ConversionFile",
    # Relational models
    "Case", "CaseName", "CaseCitationRef", "Legislation", "LegislationSection",
    "Journal", "JournalAuthor", "JournalCitationRef",
    "Treaty", "TreatyCountry", "TreatyCitationRef",
    "create_all_tables",
    "hash_password", "check_password",
    "create_user", "get_user_by_email", "set_last_login", "get_user_by_googleid",
    "save_chat_session", "get_chat_session",
    "start_session", "update_session_progress", "complete_session", "fail_session", "get_active_sessions", "get_resume_sessions", "get_session",
    "add_document", "add_embedding", "search_vector", "search_bm25", "search_hybrid", "get_file_contents",
    "add_conversion_file", "update_conversion_file_status"
]
