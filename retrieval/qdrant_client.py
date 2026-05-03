from typing import List, Dict, Optional
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from config import settings


class DenseRetriever:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    def embed_query(self, query: str) -> List[float]:
        resp = self.openai.embeddings.create(model=settings.EMBEDDING_MODEL, input=query)
        return resp.data[0].embedding

    def search(
        self,
        query: str,
        top_k: int = 20,
        ticker_filter: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        date_before: Optional[str] = None,
        date_after: Optional[str] = None,
    ) -> List[Dict]:
        conditions = []

        if ticker_filter:
            conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker_filter.upper())))

        if fiscal_year:
            conditions.append(FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year)))

        qfilter = Filter(must=conditions) if conditions else None

        results = self.qdrant.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=self.embed_query(query),
            query_filter=qfilter, limit=top_k, with_payload=True,
        )
        # print(f"  [dense] payload sample: {results[0].payload if results else None}")
        return [{**r.payload, "relevance_score": r.score} for r in results]
