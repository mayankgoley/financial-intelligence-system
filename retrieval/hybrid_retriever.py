import re
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from retrieval.qdrant_client import DenseRetriever
from retrieval.bm25_index import BM25Retriever
from retrieval.reranker import ReRanker


class HybridRetriever:
    def __init__(self):
        self.dense = DenseRetriever()
        self.bm25 = BM25Retriever()
        self.reranker = ReRanker()

    def _extract_year_from_query(self, query: str) -> Optional[int]:
        match = re.search(r'\b(20\d{2})\b', query)
        if match:
            return int(match.group(1))
        match = re.search(r'\bFY(\d{2})\b', query, re.IGNORECASE)
        if match:
            return 2000 + int(match.group(1))
        return None

    def _get_latest_fiscal_year(self, ticker: str) -> Optional[int]:
        try:
            results = self.dense.search(
                "annual revenue earnings",
                top_k=50,
                ticker_filter=ticker,
            )
            years = [
                r.get("fiscal_year")
                for r in results
                if r.get("fiscal_year") and r.get("fiscal_year") != 0
            ]
            return max(years) if years else None
        except Exception:
            return None

    def reciprocal_rank_fusion(self, ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        rrf_scores = defaultdict(float)
        chunk_map = {}
        for rlist in ranked_lists:
            for rank, chunk in enumerate(rlist):
                cid = chunk.get("chunk_id", chunk.get("text", "")[:50])
                rrf_scores[cid] += 1.0 / (k + rank)
                if cid not in chunk_map:
                    chunk_map[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [{**chunk_map[cid], "rrf_score": rrf_scores[cid]} for cid in sorted_ids]

    def _filter_by_fiscal_year(self, chunks: List[Dict], fiscal_year: int) -> List[Dict]:
        filtered = []
        removed = 0
        for chunk in chunks:
            chunk_year = chunk.get("fiscal_year", 0)
            if chunk_year == 0 or chunk_year == fiscal_year:
                filtered.append(chunk)
            else:
                removed += 1

        if removed > 0:
            print(f"  [year_filter] dropped {removed} wrong-year chunks "
                  f"(kept fy={fiscal_year} or unknown)")
        return filtered

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        ticker_filter: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        date_before: Optional[str] = None,
        date_after: Optional[str] = None,
        auto_detect_year: bool = True,
    ) -> Tuple[List[Dict], Dict]:
        timing = {}

        resolved_year = fiscal_year
        if resolved_year is None and auto_detect_year:
            resolved_year = self._extract_year_from_query(query)
            if resolved_year:
                print(f"  [auto-year] {resolved_year} from query")

        if resolved_year is None and ticker_filter:
            resolved_year = self._get_latest_fiscal_year(ticker_filter)
            if resolved_year:
                print(f"  [auto-year] no year in query, defaulting to latest fy={resolved_year} for {ticker_filter}")

        t = time.time()
        dense_results = self.dense.search(
            query,
            top_k=20,
            ticker_filter=ticker_filter,
            fiscal_year=resolved_year,
            date_before=date_before,
            date_after=date_after,
        )
        # print(f"  [dense] first hit payload: {dense_results[0] if dense_results else None}")
        if not dense_results and resolved_year:
            if ticker_filter:
                ticker_has_any_data = bool(self.dense.search(
                    query, top_k=1, ticker_filter=ticker_filter
                ))
                if ticker_has_any_data:
                    print(f"  [dense] fy={resolved_year} returned 0 but {ticker_filter} "
                          f"has data, retrying w/o year filter")
                    dense_results = self.dense.search(
                        query, top_k=20, ticker_filter=ticker_filter
                    )
                else:
                    print(f"  [dense] {ticker_filter} has no data in qdrant, not falling back")
            else:
                print(f"  [dense] fy={resolved_year} returned 0, falling back unfiltered")
                dense_results = self.dense.search(query, top_k=20)
        timing["dense_ms"] = int((time.time() - t) * 1000)

        t = time.time()
        bm25_results = self.bm25.search(
            query,
            top_k=20,
            ticker_filter=ticker_filter,
        )
        # todo: check why bm25 sometimes returns none
        timing["bm25_ms"] = int((time.time() - t) * 1000)

        t = time.time()
        fused = self.reciprocal_rank_fusion([dense_results, bm25_results])
        timing["fusion_ms"] = int((time.time() - t) * 1000)

        t = time.time()
        reranked = self.reranker.rerank(query, fused[:20], top_k=top_k)
        timing["rerank_ms"] = int((time.time() - t) * 1000)

        for c in reranked:
            c["relevance_score"] = c.get("rerank_score", 0.0)

        if resolved_year:
            t = time.time()
            reranked = self._filter_by_fiscal_year(reranked, resolved_year)
            timing["year_filter_ms"] = int((time.time() - t) * 1000)
            if len(reranked) < top_k // 2:
                print(f"  [warn] only {len(reranked)} chunks left after year filter "
                      f"(wanted {top_k})")

        timing["total_ms"] = sum(timing.values())
        print(f"  retrieved {len(reranked)} chunks in {timing['total_ms']}ms "
              f"(fy={resolved_year}, ticker={ticker_filter})")

        return reranked, timing


if __name__ == "__main__":
    r = HybridRetriever()

    print("\n=== no year filter ===")
    chunks, t = r.retrieve("What is NVIDIA's data center revenue?", top_k=5)
    for i, c in enumerate(chunks):
        print(f"  {i+1}. score={c['relevance_score']:.3f}, "
              f"fy={c.get('fiscal_year', 'N/A')}")
        print(f"     {c['text'][:200]}")

    print("\n=== explicit year ===")
    chunks, t = r.retrieve(
        "What is Netflix Q4 profit?",
        top_k=5,
        ticker_filter="NFLX",
        fiscal_year=2024,
    )
    for i, c in enumerate(chunks):
        print(f"  {i+1}. score={c['relevance_score']:.3f}, "
              f"fy={c.get('fiscal_year', 'N/A')}")
        print(f"     {c['text'][:200]}")
