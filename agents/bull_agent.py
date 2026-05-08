import re
import json
import time
from typing import Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
from agents.state import AgentState


BULL_PROMPT = """You are the BULL AGENT. Build the STRONGEST positive investment case.

RULES:
1. EVERY claim MUST cite chunk IDs: [SOURCE:chunk_id]. No citation = no claim.
2. Use ONLY numbers from the provided context. Never invent data.
3. Be specific — exact numbers, dates, quotes.
4. TEMPORAL DISCIPLINE — this is critical:
   - The query specifies a TARGET FISCAL YEAR. It will be shown to you explicitly.
   - Each chunk shows its fiscal_year in the header.
   - If a chunk's fiscal_year does NOT match the target, you MUST NOT cite it.
   - Instead, flag it as: {"wrong_period_chunk": "chunk_id", "chunk_year": X, "target_year": Y}
   - A claim using data from the wrong fiscal year is WORSE than no claim at all.
   - If no chunks match the target year, say so honestly — do not hallucinate.

OUTPUT (valid JSON only):
{
    "thesis_statement": "One-sentence bullish summary",
    "supporting_points": [
        {"claim": "...", "evidence": "...", "chunk_ids": ["..."], "confidence": 0.85}
    ],
    "risk_acknowledgments": ["..."],
    "overall_confidence": 0.8,
    "wrong_period_chunks": []
}"""


def _extract_target_year(query: str) -> Optional[int]:
    match = re.search(r'\b(20\d{2})\b', query)
    if match:
        return int(match.group(1))
    match = re.search(r'\bFY(\d{2})\b', query, re.IGNORECASE)
    if match:
        return 2000 + int(match.group(1))
    return None


def bull_agent_node(state: AgentState) -> Dict:
    start = time.time()

    target_year: Optional[int] = state.get("target_year")
    if target_year is None:
        target_year = _extract_target_year(state["query"])

    chunks_text = ""
    wrong_year_count = 0
    for c in state["context_chunks"]:
        c = c if isinstance(c, dict) else c.dict()
        chunk_fiscal_year = c.get("fiscal_year", "unknown")
        fiscal_quarter = c.get("fiscal_quarter", "")

        year_warning = ""
        if target_year and chunk_fiscal_year != "unknown" and chunk_fiscal_year != 0:
            if int(chunk_fiscal_year) != target_year:
                year_warning = f"  WARNING: fiscal_year={chunk_fiscal_year} != target={target_year} -- DO NOT CITE"
                wrong_year_count += 1

        chunks_text += (
            f"\n[CHUNK ID: {c['chunk_id']}]\n"
            f"Source: {c.get('source_doc', '')} | "
            f"fiscal_year={chunk_fiscal_year} {fiscal_quarter}"
            f"{year_warning}\n"
            f"{c['text']}\n---\n"
        )

    if wrong_year_count > 0:
        print(f"  [bull] {wrong_year_count} wrong-year chunks passed in "
              f"(target={target_year})")

    retry_note = ""
    if state.get("retry_count", 0) > 0:
        retry_note = (
            f"\n\nRETRY #{state['retry_count']}: Your previous output was flagged for "
            f"hallucinations. The most common cause is citing chunks from the wrong "
            f"fiscal year. Re-read Rule #4. Target fiscal year is {target_year}. "
            f"Any chunk with fiscal_year != {target_year} must go in wrong_period_chunks, "
            f"not in supporting_points."
        )

    year_header = (
        f"TARGET FISCAL YEAR: {target_year}\n"
        f"CRITICAL: Only cite chunks where fiscal_year == {target_year}.\n"
        f"Chunks from other years are provided for context only — do NOT cite them.\n"
    ) if target_year else (
        "TARGET FISCAL YEAR: not specified — use ONLY the most recent fiscal year "
        "present in the chunks below. Check each chunk's fiscal_year header, pick "
        "the highest year, and cite ONLY chunks from that year. Do not mix years.\n"
    )

    human_content = (
        f"QUERY: {state['query']}\n\n"
        f"{year_header}\n"
        f"MARKET DATA:\n{state.get('market_context', 'N/A')}\n\n"
        f"CHUNKS:\n{chunks_text}"
        f"{retry_note}\n\n"
        f"Return ONLY valid JSON."
    )

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=2000,
        temperature=0.3,
    )
    resp = llm.invoke([
        SystemMessage(content=BULL_PROMPT),
        HumanMessage(content=human_content),
    ])

    text = resp.content
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        thesis = json.loads(text.strip())
    except json.JSONDecodeError:
        thesis = {
            "thesis_statement": "Parse error",
            "supporting_points": [],
            "risk_acknowledgments": [],
            "overall_confidence": 0.0,
            "wrong_period_chunks": [],
            "raw": text,
        }

    agent_flagged = thesis.get("wrong_period_chunks", [])
    if agent_flagged:
        print(f"  [bull] self-flagged {len(agent_flagged)} wrong-period chunk(s)")

    ms = int((time.time() - start) * 1000)
    tokens = (
        resp.usage_metadata.get("total_tokens", 0)
        if hasattr(resp, "usage_metadata") and resp.usage_metadata
        else 0
    )

    return {
        "bull_thesis": thesis,
        "target_year": target_year,
        "token_usage": {**state.get("token_usage", {}), "bull": tokens},
        "latency_ms": {**state.get("latency_ms", {}), "bull": ms},
        "stream_events": state.get("stream_events", []) + [
            {"agent": "bull", "status": "complete", "data": thesis}
        ],
    }
