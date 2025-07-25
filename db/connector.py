"""
Database connection module for auslegalsearchv3.
- Connects to PostgreSQL with pgvector extension enabled.
- Provides SQLAlchemy engine and session makers.
- Checks/creates vector extension if needed.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_HOST = os.environ.get("AUSLEGALSEARCH_DB_HOST", "localhost")
DB_PORT = os.environ.get("AUSLEGALSEARCH_DB_PORT", "5432")
DB_USER = os.environ.get("AUSLEGALSEARCH_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("AUSLEGALSEARCH_DB_PASSWORD", "")
DB_NAME = os.environ.get("AUSLEGALSEARCH_DB_NAME", "postgres")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

from sqlalchemy.dialects.postgresql import UUID as UUIDType, JSONB
from pgvector.sqlalchemy import Vector

def ensure_pgvector():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

ensure_pgvector()
