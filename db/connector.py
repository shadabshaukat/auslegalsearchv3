"""
Database connection module for auslegalsearchv3.
- Connects to PostgreSQL with pgvector extension enabled.
- Provides SQLAlchemy engine and session makers.
- Checks/creates vector extension if needed.
"""

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Minimal .env loader (dependency-free). Reads KEY=VALUE lines and sets them in os.environ
# ONLY if the key is not already exported. This makes `source .env` (without export) work.
def _load_dotenv_file():
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        candidates = [
            os.path.abspath(os.path.join(here, "..", ".env")),   # repo root
            os.path.abspath(os.path.join(os.getcwd(), ".env")),  # current working dir
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
    except Exception:
        # Never fail due to dotenv parsing
        pass

# Load .env before reading variables (exported env vars still take precedence)
_load_dotenv_file()

DB_HOST = os.environ.get("AUSLEGALSEARCH_DB_HOST")
DB_PORT = os.environ.get("AUSLEGALSEARCH_DB_PORT")
DB_USER = os.environ.get("AUSLEGALSEARCH_DB_USER")
DB_PASSWORD = os.environ.get("AUSLEGALSEARCH_DB_PASSWORD")
DB_NAME = os.environ.get("AUSLEGALSEARCH_DB_NAME")

DB_URL = os.environ.get("AUSLEGALSEARCH_DB_URL")
if not DB_URL:
    # Require explicit env configuration to avoid accidental defaults.
    required = {
        "AUSLEGALSEARCH_DB_HOST": DB_HOST,
        "AUSLEGALSEARCH_DB_PORT": DB_PORT,
        "AUSLEGALSEARCH_DB_USER": DB_USER,
        "AUSLEGALSEARCH_DB_PASSWORD": DB_PASSWORD,
        "AUSLEGALSEARCH_DB_NAME": DB_NAME,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing required DB env vars: "
            + ", ".join(missing)
            + ". Provide AUSLEGALSEARCH_DB_URL or individual AUSLEGALSEARCH_DB_* variables (see BetaDataLoad.md)."
        )
    # Safely quote credentials in case they contain special characters like @ : / # & +
    user_q = quote_plus(DB_USER)
    pwd_q = quote_plus(DB_PASSWORD)
    DB_URL = f"postgresql+psycopg2://{user_q}:{pwd_q}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

from sqlalchemy.dialects.postgresql import UUID as UUIDType, JSONB
from pgvector.sqlalchemy import Vector

def ensure_pgvector():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
