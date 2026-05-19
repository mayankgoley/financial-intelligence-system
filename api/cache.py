import json, numpy as np
from typing import Optional
from uuid import uuid4
import redis
from openai import OpenAI
from config import settings


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.redis = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.threshold = similarity_threshold

    def _embed(self, text: str) -> np.ndarray:
        resp = self.openai.embeddings.create(model=settings.EMBEDDING_MODEL, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def get(self, query: str) -> Optional[dict]:
        # todo: O(n) scan is fine for now, replace w/ a real vector index if it grows
        query_emb = self._embed(query)
        for key in self.redis.scan_iter("cache:emb:*"):
            cached_emb = np.frombuffer(self.redis.get(key), dtype=np.float32)
            similarity = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            if similarity > self.threshold:
                result_key = key.replace(b"cache:emb:", b"cache:result:")
                cached = self.redis.get(result_key)
                if cached:
                    return json.loads(cached)
        return None

    def set(self, query: str, result: dict, ttl: int = 3600):
        key_id = uuid4().hex
        emb = self._embed(query)
        self.redis.setex(f"cache:emb:{key_id}", ttl, emb.tobytes())
        self.redis.setex(f"cache:result:{key_id}", ttl, json.dumps(result))
