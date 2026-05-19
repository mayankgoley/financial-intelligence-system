import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Text,
    DateTime, ForeignKey, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
import uuid
from config import settings

engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class QueryRun(Base):
    __tablename__ = "query_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default="running")
    total_tokens = Column(Integer, nullable=True)
    total_latency_ms = Column(Integer, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    retry_count = Column(Integer, default=0)


class AgentOutput(Base):
    __tablename__ = "agent_outputs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("query_runs.id"))
    agent_name = Column(String(50))
    output_json = Column(JSONB)
    tokens_used = Column(Integer)
    latency_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class HallucinationLog(Base):
    __tablename__ = "hallucination_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("query_runs.id"))
    claim_text = Column(Text)
    cited_chunk_id = Column(String(100))
    failure_reason = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RetrievalLog(Base):
    __tablename__ = "retrieval_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("query_runs.id"))
    chunks_returned = Column(Integer)
    bm25_latency_ms = Column(Integer)
    dense_latency_ms = Column(Integer)
    rerank_latency_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db():
    Base.metadata.create_all(bind=engine)
    print("db tables ready.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
