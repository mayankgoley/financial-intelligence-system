from typing import TypedDict, Optional, List, Dict
from pydantic import BaseModel, Field


class ChunkWithMetadata(BaseModel):
    chunk_id: str
    text: str
    source_doc: str = ""
    section: str = ""
    filing_date: str = ""
    fiscal_year: int = 0
    fiscal_quarter: str = ""
    relevance_score: float = 0.0


class HallucinationFlag(BaseModel):
    claim_text: str
    cited_chunk_id: str
    failure_reason: str
    severity: str = "high"
    chunk_fiscal_year: Optional[int] = None
    target_fiscal_year: Optional[int] = None


class AgentState(TypedDict):
    query: str
    market_context: str

    target_year: Optional[int]
    fiscal_year_filter: Optional[int]
    ticker: Optional[str]

    context_chunks: List[ChunkWithMetadata]

    bull_thesis: Optional[Dict]
    bear_thesis: Optional[Dict]
    synthesis: Optional[Dict]
    verified_output: Optional[Dict]

    hallucination_flags: List[HallucinationFlag]
    hallucination_rate: float

    retry_count: int
    token_usage: Dict[str, int]
    latency_ms: Dict[str, float]
    stream_events: List[Dict]
