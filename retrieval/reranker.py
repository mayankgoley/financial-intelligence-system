from typing import List, Dict


class ReRanker:
    def __init__(self, model_name: str = ""):
        # rrf-only for now, CrossEncoder import was killing startup time
        # todo: plug in a real cross-encoder once we have a gpu box
        pass

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 8) -> List[Dict]:
        if not chunks:
            return []

        for c in chunks:
            c["rerank_score"] = c.get("rrf_score", c.get("relevance_score", c.get("bm25_score", 0.0)))

        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]
