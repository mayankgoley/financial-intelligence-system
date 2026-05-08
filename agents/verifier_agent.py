import json
import time
from typing import Dict, List, Set
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
from agents.state import AgentState, HallucinationFlag


VERIFY_PROMPT = """You are the VERIFIER. Your ONLY job is to check if each claim's
specific data is actually present in its cited chunk. You are NOT re-running
adversarial analysis or judging whether the reasoning is strong.

STRICT DEFINITIONS:

  not_grounded: The specific number, fact, date, or metric in the claim does NOT
    appear anywhere in the cited chunk. The chunk simply doesn't contain the data.
    Example: claim says "revenue grew 40%" but chunk never mentions 40%.

  over_extrapolated: The data IS in the chunk, but the claim adds evaluative
    words the chunk doesn't use — words like "leader", "dominant", "best",
    "exceptional", "proves". The underlying fact is real, just overstated.

  grounded: The claim's core data point is directly stated or clearly present
    in the chunk as a number, quote, or explicit fact.

DO NOT flag these — they are reasoning issues, not data issues:
  - The Bull's argument is one-sided or incomplete
  - Important risks or context are missing
  - The logical conclusion doesn't follow from the data
  - A competitor comparison is absent

Only flag what is LITERALLY not in the chunk.

OUTPUT (valid JSON only):
{
    "verified_claims": [
        {
            "claim": "...",
            "chunk_ids": ["..."],
            "status": "verified|flagged",
            "reason": "grounded|not_grounded|over_extrapolated|wrong_fiscal_year",
            "explanation": "Quote the specific text that grounds or disproves the claim"
        }
    ],
    "summary": "One sentence on data integrity"
}"""


TRUE_HALLUCINATION_TYPES = {"wrong_fiscal_year", "chunk_not_found", "not_grounded"}
REASONING_WEAKNESS_TYPES = {"over_extrapolated", "logical_flaw", "missing_context", "unverifiable"}


def _is_true_hallucination(failure_reason: str) -> bool:
    return failure_reason in TRUE_HALLUCINATION_TYPES


def _severity_for_reason(reason: str) -> str:
    if reason in ("wrong_fiscal_year", "chunk_not_found"):
        return "high"
    if reason == "not_grounded":
        return "medium"
    return "low"


def _build_chunk_lookup(chunks) -> Dict[str, Dict]:
    lookup = {}
    for c in chunks:
        c = c if isinstance(c, dict) else c.dict()
        lookup[c["chunk_id"]] = c
    return lookup


def _get_flag_attr(flag, attr: str, default=""):
    return getattr(flag, attr, None) or (flag.get(attr, default) if isinstance(flag, dict) else default)


def _stage0_temporal_check(
    claims: List[Dict],
    chunk_lookup: Dict[str, Dict],
    target_year: int,
) -> List[HallucinationFlag]:
    flags = []
    for c in claims:
        if c.get("stage1_fail"):
            continue
        chunk = chunk_lookup.get(c["chunk_id"], {})
        chunk_year = chunk.get("fiscal_year", 0)
        if chunk_year == 0:
            # todo: backfill fiscal_year for old chunks, this silently bypasses
            continue
        if chunk_year != target_year:
            flags.append(HallucinationFlag(
                claim_text=c["claim"],
                cited_chunk_id=c["chunk_id"],
                failure_reason="wrong_fiscal_year",
                severity="high",
                chunk_fiscal_year=chunk_year,
                target_fiscal_year=target_year,
            ))
            print(f"  [stage0] flag {c['chunk_id']} "
                  f"fy={chunk_year} != target={target_year}")
    return flags


def verifier_agent_node(state: AgentState) -> Dict:
    start = time.time()
    synthesis = state.get("synthesis", {})
    chunks = state["context_chunks"]
    target_year = state.get("target_year")

    chunk_lookup = _build_chunk_lookup(chunks)

    claims = []
    for kp in synthesis.get("key_points", []):
        claim_text = kp.get("point", "")
        for cid in kp.get("chunk_ids", []):
            if cid not in chunk_lookup:
                claims.append({
                    "claim": claim_text, "chunk_id": cid,
                    "chunk_text": "[DOES NOT EXIST]", "stage1_fail": True,
                })
            else:
                claims.append({
                    "claim": claim_text, "chunk_id": cid,
                    "chunk_text": chunk_lookup[cid]["text"], "stage1_fail": False,
                })

    if not claims:
        return {
            "verified_output": synthesis,
            "hallucination_flags": [],
            "hallucination_rate": 0.0,
            "stream_events": state.get("stream_events", []) + [
                {"agent": "verifier", "status": "complete", "data": {"rate": 0.0}}
            ],
        }

    unique_claim_texts: Set[str] = {c["claim"] for c in claims}
    total_unique_claims = max(len(unique_claim_texts), 1)

    all_flags: List[HallucinationFlag] = []

    if target_year:
        all_flags.extend(_stage0_temporal_check(claims, chunk_lookup, target_year))

    already_flagged = {_get_flag_attr(f, "claim_text") for f in all_flags}
    for c in claims:
        if c["stage1_fail"] and c["claim"] not in already_flagged:
            all_flags.append(HallucinationFlag(
                claim_text=c["claim"],
                cited_chunk_id=c["chunk_id"],
                failure_reason="chunk_not_found",
                severity="high",
            ))
            already_flagged.add(c["claim"])

    already_flagged = {_get_flag_attr(f, "claim_text") for f in all_flags}
    claims_for_llm = [
        c for c in claims
        if not c["stage1_fail"] and c["claim"] not in already_flagged
    ]

    tokens = 0
    if claims_for_llm:
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=settings.ANTHROPIC_API_KEY,
            max_tokens=2000,
            temperature=0.1,
        )

        year_context = (
            f"TARGET FISCAL YEAR: {target_year}. "
            f"Flag wrong_fiscal_year if the claim explicitly states a different year.\n\n"
            if target_year else ""
        )

        claims_text = year_context
        for i, c in enumerate(claims_for_llm):
            chunk = chunk_lookup.get(c["chunk_id"], {})
            chunk_year = chunk.get("fiscal_year", "unknown")
            claims_text += (
                f"\nCLAIM {i+1}: {c['claim']}\n"
                f"CITED CHUNK [{c['chunk_id']}] (fiscal_year={chunk_year}):\n"
                f"{c['chunk_text'][:500]}\n---\n"
            )

        resp = llm.invoke([
            SystemMessage(content=VERIFY_PROMPT),
            HumanMessage(content=f"Check data grounding only:\n{claims_text}\nReturn ONLY valid JSON."),
        ])

        text = resp.content
        try:
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            verification = json.loads(text.strip())
        except json.JSONDecodeError:
            verification = {"verified_claims": []}

        for vc in verification.get("verified_claims", []):
            if vc.get("status") == "flagged":
                reason = vc.get("reason", "not_grounded")
                all_flags.append(HallucinationFlag(
                    claim_text=vc.get("claim", ""),
                    cited_chunk_id=(
                        vc.get("chunk_ids", ["unknown"])[0]
                        if vc.get("chunk_ids") else "unknown"
                    ),
                    failure_reason=reason,
                    severity=_severity_for_reason(reason),
                ))

        tokens = (
            resp.usage_metadata.get("total_tokens", 0)
            if hasattr(resp, "usage_metadata") and resp.usage_metadata else 0
        )
    else:
        print("  [verifier] no claims left after stage 0/1, skipping llm")

    true_hallucination_flags = [
        f for f in all_flags
        if _is_true_hallucination(_get_flag_attr(f, "failure_reason"))
    ]
    reasoning_weakness_flags = [
        f for f in all_flags
        if not _is_true_hallucination(_get_flag_attr(f, "failure_reason"))
    ]

    unique_hallucinated = {_get_flag_attr(f, "claim_text") for f in true_hallucination_flags}
    hallucination_rate = len(unique_hallucinated) / total_unique_claims

    unique_weak = {_get_flag_attr(f, "claim_text") for f in reasoning_weakness_flags}
    reasoning_weakness_rate = len(unique_weak) / total_unique_claims

    stripped_claims = {
        _get_flag_attr(f, "claim_text") for f in true_hallucination_flags
        if _get_flag_attr(f, "severity") == "high"
    }
    verified = synthesis.copy()
    verified["key_points"] = [
        kp for kp in verified.get("key_points", [])
        if kp.get("point") not in stripped_claims
    ]
    verified["hallucination_rate"] = hallucination_rate
    verified["reasoning_weakness_rate"] = reasoning_weakness_rate
    verified["flags_count"] = len(true_hallucination_flags)
    verified["reasoning_flags_count"] = len(reasoning_weakness_flags)

    verified["flags_by_reason"] = {
        r: sum(1 for f in all_flags if _get_flag_attr(f, "failure_reason") == r)
        for r in ("wrong_fiscal_year", "chunk_not_found", "not_grounded", "over_extrapolated")
    }
    verified["flags_by_severity"] = {
        "high": sum(1 for f in true_hallucination_flags if _get_flag_attr(f, "severity") == "high"),
        "medium": sum(1 for f in true_hallucination_flags if _get_flag_attr(f, "severity") == "medium"),
        "low": len(reasoning_weakness_flags),
    }

    ms = int((time.time() - start) * 1000)
    print(f"  [verifier] hall={hallucination_rate:.0%} | "
          f"weak={reasoning_weakness_rate:.0%} | "
          f"true_flags={len(true_hallucination_flags)} | "
          f"weak_flags={len(reasoning_weakness_flags)} | "
          f"claims={total_unique_claims}")

    return {
        "verified_output": verified,
        "hallucination_flags": true_hallucination_flags,
        "hallucination_rate": hallucination_rate,
        "token_usage": {**state.get("token_usage", {}), "verifier": tokens},
        "latency_ms": {**state.get("latency_ms", {}), "verifier": ms},
        "stream_events": state.get("stream_events", []) + [{
            "agent": "verifier",
            "status": "complete",
            "data": {
                "rate": hallucination_rate,
                "reasoning_weakness_rate": reasoning_weakness_rate,
                "flags": len(true_hallucination_flags),
                "reasoning_flags": len(reasoning_weakness_flags),
                "by_reason": verified["flags_by_reason"],
                "by_severity": verified["flags_by_severity"],
            },
        }],
    }
