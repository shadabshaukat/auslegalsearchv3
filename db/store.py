"""
Centralized DB models and ORM for auslegalsearchv3.
- Exports all tables, full CRUD/session logic for users, ingestion, search, embedding, chat, and conversion files.
- All app code imports models and functions from here, schema created with create_all_tables().
"""

from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, Float
from sqlalchemy import select, desc, text
from db.connector import engine, SessionLocal, Vector, JSONB, UUIDType
from embedding.embedder import Embedder
from datetime import datetime
import uuid
import os
import bcrypt

embedder = Embedder()
EMBEDDING_DIM = embedder.dimension

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

def create_all_tables():
    Base.metadata.create_all(engine)

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
        SELECT embeddings.doc_id, embeddings.chunk_index, embeddings.vector <#> {array_sql} AS score,
            documents.content, documents.source, documents.format, embeddings.chunk_metadata
        FROM embeddings
        JOIN documents ON embeddings.doc_id = documents.id
        ORDER BY embeddings.vector <#> {array_sql} ASC
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

create_all_tables()

__all__ = [
    "Base", "engine", "SessionLocal", "Vector", "JSONB", "UUIDType",
    "User", "Document", "Embedding", "EmbeddingSession", "EmbeddingSessionFile", "ChatSession", "ConversionFile",
    "create_all_tables",
    "hash_password", "check_password",
    "create_user", "get_user_by_email", "set_last_login", "get_user_by_googleid",
    "save_chat_session", "get_chat_session",
    "start_session", "update_session_progress", "complete_session", "fail_session", "get_active_sessions", "get_resume_sessions", "get_session",
    "add_document", "add_embedding", "search_vector", "search_bm25", "search_hybrid", "get_file_contents",
    "add_conversion_file", "update_conversion_file_status"
]
