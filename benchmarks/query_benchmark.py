import json, time
from agents.graph import agent_graph
from retrieval.hybrid_retriever import HybridRetriever
from ingestion.market_data import format_market_context

# manual benchmark set, not great coverage but it's what I had
BENCHMARK_QUERIES = [
    {"query": "Analyze NVIDIA's data center revenue growth", "ticker": "NVDA"},
    {"query": "What are Apple's main risk factors?", "ticker": "AAPL"},
    {"query": "Compare Microsoft's cloud revenue trajectory", "ticker": "MSFT"},
    {"query": "What is NVIDIA's gross margin trend?", "ticker": "NVDA"},
    {"query": "Evaluate Apple's services segment growth", "ticker": "AAPL"},
]


def run_benchmark():
    retriever = HybridRetriever()
    results = []

    for i, bq in enumerate(BENCHMARK_QUERIES):
        print(f"\nq {i+1}/{len(BENCHMARK_QUERIES)}: {bq['query']}")
        start = time.time()

        chunks, timing = retriever.retrieve(bq["query"], top_k=8, ticker_filter=bq["ticker"])
        market = format_market_context(bq["ticker"])

        state = {
            "query": bq["query"], "market_context": market,
            "context_chunks": chunks, "bull_thesis": None, "bear_thesis": None,
            "synthesis": None, "verified_output": None,
            "hallucination_flags": [], "hallucination_rate": 0.0,
            "retry_count": 0, "token_usage": {}, "latency_ms": {}, "stream_events": [],
        }

        result = agent_graph.invoke(state)
        total_ms = int((time.time()-start)*1000)

        results.append({
            "query": bq["query"],
            "hallucination_rate": result.get("hallucination_rate", 0),
            "total_ms": total_ms,
            "token_usage": result.get("token_usage", {}),
            "retry_count": result.get("retry_count", 0),
        })
        print(f"  {total_ms}ms | hall: {result.get('hallucination_rate',0):.1%}")

    avg_hall = sum(r["hallucination_rate"] for r in results) / len(results)
    avg_ms = sum(r["total_ms"] for r in results) / len(results)
    print(f"\n=== results ({len(results)} queries) ===")
    print(f"  avg hall: {avg_hall:.1%}")
    print(f"  avg ms: {avg_ms:.0f}")

    with open("docs/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_benchmark()
