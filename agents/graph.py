import re
from typing import Optional
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.bull_agent import bull_agent_node
from agents.bear_agent import bear_agent_node
from agents.synthesis_agent import synthesis_agent_node
from agents.verifier_agent import verifier_agent_node
from retrieval.hybrid_retriever import HybridRetriever

_retriever = None

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def should_retry(state: AgentState) -> str:
    rate = state.get("hallucination_rate", 0.0)
    retries = state.get("retry_count", 0)

    # print(f"  [graph] state keys: {list(state.keys())}")

    if retries >= 2:
        # hack: hardcoded cap, 2 felt right
        print(f"  [graph] max retries ({retries}), giving up")
        return "done"

    flags = state.get("hallucination_flags", [])
    temporal_flags = [
        f for f in flags
        if (f.failure_reason if hasattr(f, "failure_reason") else f.get("failure_reason", ""))
        == "wrong_fiscal_year"
    ]

    if temporal_flags:
        print(f"  [graph] {len(temporal_flags)} wrong-year flag(s), retrying "
              f"({retries + 1}/2)")
        return "retry"

    if rate > 0.30:
        # todo: 0.30 is a guess, should tune this on real queries
        print(f"  [graph] rate {rate:.0%} > 30%, retry {retries + 1}/2")
        return "retry"

    print(f"  [graph] rate {rate:.0%} ok, done")
    return "done"


def smart_retry(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0) + 1

    target_year: Optional[int] = state.get("target_year")
    if target_year is None:
        match = re.search(r'\b(20\d{2})\b', state.get("query", ""))
        if match:
            target_year = int(match.group(1))
        else:
            match = re.search(r'\bFY(\d{2})\b', state.get("query", ""), re.IGNORECASE)
            if match:
                target_year = 2000 + int(match.group(1))

    ticker = state.get("ticker")

    if target_year:
        print(f"  [smart_retry] year={target_year} ticker={ticker} "
              f"({retry_count}/2)")
        try:
            new_chunks, timing = _get_retriever().retrieve(
                query=state["query"],
                top_k=8,
                ticker_filter=ticker,
                fiscal_year=target_year,
                auto_detect_year=False,
            )
            # print(f"  [smart_retry] timing={timing}")
            print(f"  [smart_retry] got {len(new_chunks)} chunks "
                  f"in {timing.get('total_ms', '?')}ms")

            if not new_chunks:
                print(f"  [smart_retry] nothing for fy={target_year}, keeping old chunks")
                new_chunks = state["context_chunks"]

        except Exception as e:
            print(f"  [smart_retry] retrieval blew up: {e}, keeping old chunks")
            new_chunks = state["context_chunks"]
    else:
        print(f"  [smart_retry] no year found, keeping old chunks ({retry_count}/2)")
        new_chunks = state["context_chunks"]

    return {
        "retry_count": retry_count,
        "context_chunks": new_chunks,
        "target_year": target_year,
        "fiscal_year_filter": target_year,
        "bull_thesis": None,
        "bear_thesis": None,
        "synthesis": None,
        "hallucination_flags": [],
        "hallucination_rate": 0.0,
    }


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("bull", bull_agent_node)
    graph.add_node("bear", bear_agent_node)
    graph.add_node("synthesize", synthesis_agent_node)
    graph.add_node("verifier", verifier_agent_node)
    graph.add_node("smart_retry", smart_retry)

    graph.set_entry_point("bull")
    graph.add_edge("bull", "bear")
    graph.add_edge("bear", "synthesize")
    graph.add_edge("synthesize", "verifier")

    graph.add_conditional_edges(
        "verifier",
        should_retry,
        {"retry": "smart_retry", "done": END},
    )

    graph.add_edge("smart_retry", "bull")

    return graph.compile()


agent_graph = build_graph()
