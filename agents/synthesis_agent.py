import json, time
from typing import Dict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
from agents.state import AgentState

SYNTHESIS_PROMPT = """You are the SYNTHESIS AGENT. Weigh Bull vs Bear and produce a balanced thesis.

GUIDELINES:
1. Accept well-grounded Bear challenges. Dismiss speculative ones.
2. Preserve ALL chunk_id citations.
3. Give clear recommendation: BULLISH, BEARISH, or NEUTRAL.

OUTPUT (valid JSON):
{
    "recommendation": "BULLISH|BEARISH|NEUTRAL",
    "confidence": 0.75,
    "thesis_summary": "2-3 sentence balanced summary",
    "key_points": [{"point": "...", "chunk_ids": ["..."], "bull_bear_alignment": "..."}],
    "scenarios": {
        "bull_case": {"description": "...", "probability": 0.3},
        "base_case": {"description": "...", "probability": 0.5},
        "bear_case": {"description": "...", "probability": 0.2}
    },
    "key_risks": ["..."], "key_catalysts": ["..."]
}"""

def synthesis_agent_node(state: AgentState) -> Dict:
    start = time.time()
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=settings.ANTHROPIC_API_KEY, max_tokens=2500, temperature=0.3)
    resp = llm.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=f"QUERY: {state['query']}\n\nMARKET: {state.get('market_context','')}\n\nBULL:\n{json.dumps(state.get('bull_thesis',{}),indent=2)}\n\nBEAR:\n{json.dumps(state.get('bear_thesis',{}),indent=2)}\n\nReturn ONLY valid JSON."),
    ])
    text = resp.content
    try:
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        elif "```" in text: text = text.split("```")[1].split("```")[0]
        synth = json.loads(text.strip())
    except json.JSONDecodeError:
        synth = {"recommendation":"NEUTRAL","confidence":0.0,"thesis_summary":"Parse error","key_points":[],"raw":text}

    ms = int((time.time()-start)*1000)
    tokens = resp.usage_metadata.get("total_tokens",0) if hasattr(resp,'usage_metadata') and resp.usage_metadata else 0
    return {
        "synthesis": synth,
        "token_usage": {**state.get("token_usage",{}), "synthesis": tokens},
        "latency_ms": {**state.get("latency_ms",{}), "synthesis": ms},
        "stream_events": state.get("stream_events",[]) + [{"agent":"synthesis","status":"complete","data":synth}],
    }
