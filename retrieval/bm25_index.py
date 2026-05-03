import pickle
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        # hack: pickle on disk, fine until two ingestion runs race
        self.bm25 = None
        self.chunks = []
        path = Path(index_path)
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["index"]
            self.chunks = data["chunks"]
            print(f"bm25: {len(self.chunks)} chunks loaded")

    def search(self, query: str, top_k: int = 20, ticker_filter: Optional[str] = None) -> List[Dict]:
        if not self.bm25: return []
        scores = self.bm25.get_scores(query.lower().split())
        # print(f"  [bm25] top score: {max(scores) if len(scores) else 'n/a'}")
        scored = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        if ticker_filter:
            scored = [(c,s) for c,s in scored if c.get("ticker","").upper() == ticker_filter.upper()]
        return [{**c, "bm25_score": float(s)} for c,s in scored[:top_k] if s > 0]
