import json
import time
from typing import Dict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
from agents.state import AgentState


BEAR_PROMPT = """You are the BEAR AGENT. Challenge EVERY claim in the Bull thesis.

For each claim, do ONE of:
1. CONTRADICT:    Find opposing evidence in the chunks (cite chunk_ids).
2. LOGICAL_FLAW:  Identify reasoning errors or unsupported conclusions.
3. MISSING_CONTEXT: Point out important information the Bull ignored.
4. UNVERIFIABLE:  Flag claims that can't be verified from the provided chunks.
5. WRONG_PERIOD:  Flag claims that cite data from the wrong fiscal year.
   - Check the fiscal_year shown in each chunk header.
   - If the Bull cited a chunk whose fiscal_year != TARGET FISCAL YEAR, this is
     a WRONG_PERIOD challenge with severity="high".

OUTPUT (valid JSON only):
{
    "overall_assessment": "Summary of Bull thesis weaknesses",
    "challenges": [
        {
            "original_claim": "...",
            "challenge": "...",
            "challenge_type": "contradiction|logical_flaw|missing_context|unverifiable|wrong_period",
            "supporting_chunk_ids": [],
            "severity": "high|medium|low"
        }
    ],
    "strongest_bull_point": "Which Bull claim is actually well-supported",
    "biggest_risk": "The most important risk the Bull thesis ignores"
}"""


def bear_agent_node(state: AgentState) -> Dict:
    start = time.time()

    target_year = state.get("target_year")
    chunks_text = ""
    for c in state["context_chunks"]:
        c = c if isinstance(c, dict) else c.dict()
        chunk_fiscal_year = c.get("fiscal_year", "unknown")
        fiscal_quarter = c.get("fiscal_quarter", "")

        year_warning = ""
        if target_year and chunk_fiscal_year != "unknown" and chunk_fiscal_year != 0:
            if int(chunk_fiscal_year) != target_year:
                year_warning = f"  WRONG PERIOD: fiscal_year={chunk_fiscal_year} != target={target_year}"

        chunks_text += (
            f"\n[CHUNK ID: {c['chunk_id']}]\n"
            f"fiscal_year={chunk_fiscal_year} {fiscal_quarter}"
            f"{year_warning}\n"
            f"{c['text']}\n---\n"
        )

    bull_thesis = state.get("bull_thesis", {})
    wrong_period_note = ""
    bull_wrong_period = bull_thesis.get("wrong_period_chunks", [])
    if bull_wrong_period:
        wrong_period_note = (
            f"\n\nNOTE: The Bull Agent self-identified {len(bull_wrong_period)} "
            f"wrong-period chunk(s): {bull_wrong_period}. "
            f"These should be challenged as WRONG_PERIOD with severity=high."
        )

    year_context = (
        f"TARGET FISCAL YEAR: {target_year}\n"
        f"Any Bull claim citing a chunk with fiscal_year != {target_year} "
        f"must be challenged as WRONG_PERIOD.\n\n"
    ) if target_year else ""

    human_content = (
        f"QUERY: {state['query']}\n\n"
        f"{year_context}"
        f"BULL THESIS:\n{json.dumps(bull_thesis, indent=2)}"
        f"{wrong_period_note}\n\n"
        f"CHUNKS:\n{chunks_text}\n\n"
        f"Return ONLY valid JSON."
    )

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=2000,
        temperature=0.4,
    )
    resp = llm.invoke([
        SystemMessage(content=BEAR_PROMPT),
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
            "overall_assessment": "Parse error",
            "challenges": [],
            "strongest_bull_point": "",
            "biggest_risk": "",
            "raw": text,
        }

    wrong_period_challenges = [
        ch for ch in thesis.get("challenges", [])
        if ch.get("challenge_type") == "wrong_period"
    ]
    if wrong_period_challenges:
        print(f"  [bear] {len(wrong_period_challenges)} wrong_period challenge(s)")

    ms = int((time.time() - start) * 1000)
    tokens = (
        resp.usage_metadata.get("total_tokens", 0)
        if hasattr(resp, "usage_metadata") and resp.usage_metadata
        else 0
    )

    return {
        "bear_thesis": thesis,
        "token_usage": {**state.get("token_usage", {}), "bear": tokens},
        "latency_ms": {**state.get("latency_ms", {}), "bear": ms},
        "stream_events": state.get("stream_events", []) + [
            {"agent": "bear", "status": "complete", "data": thesis}
        ],
    }
