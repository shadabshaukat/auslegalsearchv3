"""
FastAPI backend for AUSLegalSearchv3, with legal-tuned hybrid QA, legal-aware chunking, RAG, reranker interface, and full system prompt config.
"""

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os
import secrets
import json
import inspect

from db.store import (
    create_user, get_user_by_email, hash_password, check_password,
    add_document, add_embedding, start_session, get_session, complete_session, fail_session, update_session_progress,
    search_vector, search_bm25, search_hybrid, get_chat_session, save_chat_session,
    get_active_sessions, get_resume_sessions,
)
from embedding.embedder import Embedder
from rag.rag_pipeline import RAGPipeline, list_ollama_models
from ingest.loader import walk_legal_files, parse_txt, parse_html, chunk_document

RERANKER_DATA_PATH = "./reranker_models.json"

_DEFAULT_RERANKER_MODELS = [
    {"name": "mxbai-rerank-xsmall", "desc": "General XC; top-3 HuggingFace scorer", "hf": "mixedbread-ai/mxbai-rerank-xsmall"},
    {"name": "ms-marco-MiniLM-L-6-v2", "desc": "MS MARCO cross-encoder baseline", "hf": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    {"name": "nlpaueb-legal-bert-small-uncased", "desc": "Legal-domain (smaller) classifier", "hf": "nlpaueb/legal-bert-small-uncased"},
]

def load_reranker_models():
    if os.path.exists(RERANKER_DATA_PATH):
        with open(RERANKER_DATA_PATH, "r") as f:
            data = json.load(f)
            return {m["name"]: m for m in data}
    return {m["name"]: m for m in _DEFAULT_RERANKER_MODELS}

def save_reranker_models(registry):
    with open(RERANKER_DATA_PATH, "w") as f:
        json.dump(list(registry.values()), f, indent=2)

_INSTALLED_RERANKERS = load_reranker_models()
_DEFAULT_MODEL = "mxbai-rerank-xsmall"

def available_rerankers():
    return list(_INSTALLED_RERANKERS.values())

def get_reranker_model(model_name):
    if not model_name or model_name not in _INSTALLED_RERANKERS:
        model_name = _DEFAULT_MODEL
    m = _INSTALLED_RERANKERS[model_name]
    return m

def download_hf_model(hf_repo):
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder(hf_repo)
    except Exception as e:
        print(f"Could not download {hf_repo}: {e}")

app = FastAPI(
    title="AUSLegalSearchv3 API",
    description="REST API for legal vector search, ingestion, RAG, chat, reranker, and system prompt management.",
    version="0.30"
)

security = HTTPBasic()
def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    api_user = os.environ.get("FASTAPI_API_USER", "legal_api")
    api_pass = os.environ.get("FASTAPI_API_PASS", "letmein")
    correct_username = secrets.compare_digest(credentials.username, api_user)
    correct_password = secrets.compare_digest(credentials.password, api_pass)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect API credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- User endpoints ---
class UserCreateReq(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

@app.post("/users", tags=["users"])
def api_create_user(user: UserCreateReq, _: str = Depends(get_current_user)):
    db_user = get_user_by_email(user.email)
    if db_user:
        raise HTTPException(400, detail="Email already registered")
    return create_user(email=user.email, password=user.password, name=user.name)

# --- Ingestion endpoints ---
class IngestStartReq(BaseModel):
    directory: str
    session_name: str

@app.post("/ingest/start", tags=["ingest"])
def api_ingest_start(req: IngestStartReq, _: str = Depends(get_current_user)):
    session = start_session(session_name=req.session_name, directory=req.directory)
    return {"session_name": session.session_name, "status": session.status, "started_at": session.started_at}

@app.get("/ingest/sessions", tags=["ingest"])
def api_active_ingest_sessions(_: str = Depends(get_current_user)):
    return [s.session_name for s in get_active_sessions()]

@app.post("/ingest/stop", tags=["ingest"])
def api_stop_ingest(session_name: str, _: str = Depends(get_current_user)):
    fail_session(session_name)
    return {"session": session_name, "status": "stopped"}

# --- Document endpoints ---
@app.get("/documents", tags=["documents"])
def api_list_documents(_: str = Depends(get_current_user)):
    from db.store import SessionLocal, Document
    with SessionLocal() as session:
        docs = session.query(Document).limit(100).all()
        return [{"id": d.id, "source": d.source, "format": d.format} for d in docs]

@app.get("/documents/{doc_id}", tags=["documents"])
def api_get_document(doc_id: int, _: str = Depends(get_current_user)):
    from db.store import SessionLocal, Document
    with SessionLocal() as session:
        d = session.query(Document).filter_by(id=doc_id).first()
        if not d:
            raise HTTPException(404, "Document not found")
        return {"id": d.id, "source": d.source, "content": d.content, "format": d.format}

# --- Embedding Search endpoints ---
class SearchReq(BaseModel):
    query: str
    top_k: int = 5
    model: Optional[str] = None

@app.post("/search/vector", tags=["search"])
def api_search_vector(req: SearchReq, _: str = Depends(get_current_user)):
    embedder = Embedder()
    query_vec = embedder.embed([req.query])[0]
    hits = search_vector(query_vec, top_k=req.top_k)
    # Include metadata in results
    for hit in hits:
        if "chunk_metadata" not in hit:
            hit["chunk_metadata"] = None
    return hits

@app.post("/search/rerank", tags=["search"])
def api_search_rerank(req: SearchReq, _: str = Depends(get_current_user)):
    reranker_info = get_reranker_model(req.model)
    embedder = Embedder()
    query_vec = embedder.embed([req.query])[0]
    hits = search_vector(query_vec, top_k=max(20, req.top_k))
    hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)[:req.top_k]
    for h in hits:
        h["reranker"] = reranker_info["name"]
        if "chunk_metadata" not in h:
            h["chunk_metadata"] = None
    return hits

@app.get("/models/reranker", tags=["models"])
def api_reranker_models(_: str = Depends(get_current_user)):
    return available_rerankers()

@app.get("/models/rerankers", tags=["models"])
def api_rerankers_list(_: str = Depends(get_current_user)):
    # Return just a list of installed reranker names for dropdowns etc.
    return list(_INSTALLED_RERANKERS.keys())

class RerankerDownloadReq(BaseModel):
    name: str
    hf_repo: str
    desc: Optional[str] = None

@app.post("/models/reranker/download", tags=["models"])
def download_reranker_model(req: RerankerDownloadReq, background_tasks: BackgroundTasks, _: str = Depends(get_current_user)):
    if req.name in _INSTALLED_RERANKERS:
        return {"message": f"Model {req.name} is already installed.", "models": available_rerankers()}
    background_tasks.add_task(download_hf_model, req.hf_repo)
    entry = {"name": req.name, "desc": req.desc or "", "hf": req.hf_repo}
    _INSTALLED_RERANKERS[req.name] = entry
    save_reranker_models(_INSTALLED_RERANKERS)
    return {"message": f"Download started for {req.name}", "models": available_rerankers()}

# --- Hybrid Legal Search endpoint ---
class HybridSearchReq(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5

@app.post("/search/hybrid", tags=["search"])
def api_search_hybrid(req: HybridSearchReq, _: str = Depends(get_current_user)):
    """
    Hybrid (vector+bm25) legal search with score and citation output, now with metadata.
    """
    results = search_hybrid(req.query, top_k=req.top_k, alpha=req.alpha)
    fields = ["doc_id", "chunk_index", "hybrid_score", "citation", "score", "vector_score", "bm25_score", "text", "source", "format", "chunk_metadata"]
    filtered = [
        {k: r[k] for k in fields if k in r}
        for r in results
    ]
    return filtered

# --- RAG/Llama Legal QA with memory ---
class RagReq(BaseModel):
    question: str
    context_chunks: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    chunk_metadata: Optional[List[Dict[str, Any]]] = None
    custom_prompt: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.90
    max_tokens: int = 1024
    repeat_penalty: float = 1.1
    model: Optional[str] = "llama3"
    reranker_model: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

@app.post("/search/rag", tags=["search"])
def api_search_rag(req: RagReq, _: str = Depends(get_current_user)):
    rag = RAGPipeline(model=req.model or "llama3")
    query_kwargs = dict(
        context_chunks=req.context_chunks,
        sources=req.sources,
        chunk_metadata=req.chunk_metadata,
        custom_prompt=req.custom_prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        repeat_penalty=req.repeat_penalty,
    )
    query_sig = inspect.signature(rag.query)
    if req.chat_history is not None and "chat_history" in query_sig.parameters:
        query_kwargs["chat_history"] = req.chat_history
    return rag.query(
        req.question,
        **query_kwargs
    )

# --- Chat Session endpoints ---
class ChatMsg(BaseModel):
    prompt: str
    model: Optional[str] = "llama3"

@app.post("/chat/session", tags=["chat"])
def api_chat_session(msg: ChatMsg, _: str = Depends(get_current_user)):
    rag = RAGPipeline(model=msg.model or "llama3")
    results = search_bm25(msg.prompt, top_k=5)
    context_chunks = [r["text"] for r in results]
    sources = [r["source"] for r in results]
    chunk_metadata = [r.get("chunk_metadata") for r in results]
    answer = rag.query(
        msg.prompt, context_chunks=context_chunks, sources=sources, chunk_metadata=chunk_metadata
    )["answer"]
    return {
        "answer": answer,
        "sources": sources,
        "chunk_metadata": chunk_metadata
    }

# --- Utility/model endpoints ---
@app.get("/models/ollama", tags=["models"])
def api_ollama_models(_: str = Depends(get_current_user)):
    return list_ollama_models()

@app.get("/files/ls", tags=["files"])
def api_ls(path: str, _: str = Depends(get_current_user)):
    allowed_roots = [os.getcwd(), "/home/ubuntu/data"]
    real = os.path.realpath(path)
    if not any(real.startswith(os.path.realpath(x)) for x in allowed_roots):
        raise HTTPException(403, "Not allowed")
    if os.path.isdir(real):
        return os.listdir(real)
    else:
        return [real]

@app.get("/health", tags=["utility"])
def healthcheck():
    return {"ok": True}
