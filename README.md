# financial-intelligence-system

Multi-agent bull/bear/synthesis debate over SEC filings with a per-claim hallucination check and a live market-data sidecar.

## architecture / stack

- ingestion: pulls 10-K / 10-Q from SEC EDGAR, chunks via langchain splitter, embeds with `text-embedding-3-large`, indexes into Qdrant + a pickled BM25.
- retrieval: hybrid dense + BM25 with RRF fusion, plus a post-rerank fiscal-year filter.
- agents: LangGraph state machine — bull → bear → synthesis → verifier — with a conditional retry edge that re-pulls chunks under a strict year filter if the verifier flags hallucinations.
- api: FastAPI + SSE, Jinja2 templates, Tailwind via CDN.
- python 3.12 · FastAPI · LangGraph · LangChain
- OpenAI (embeddings) · Anthropic Claude (agents — sonnet for bull/synth/verify, haiku for bear)
- Qdrant (vectors) · BM25 (keyword) · Postgres (run logs) · Redis (semantic cache)
- yfinance (live market data)
- docker compose for local infra

## local setup

```bash
cp .env.example .env
# fill OPENAI_API_KEY and ANTHROPIC_API_KEY

docker compose up -d           # postgres, redis, qdrant, kafka, zookeeper
pip install -r requirements.txt

python -m api.database         # one-time, creates tables

uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000 and submit a query. First hit per ticker auto-ingests filings from EDGAR (~30-60s); subsequent queries hit Qdrant directly.

Postgres is mapped to host port `5433` (port 5432 was taken on the dev box). The `.env` matches.

## known issues / todo

- SEC EDGAR rate limit is annoying. Currently a flat `time.sleep(0.15)` between requests in `EdgarClient._rate_limit`. Fine for serial ingestion, gets 403'd if you parallelize. Should swap in a token bucket.
- LangGraph retry edge sometimes loops too many times — when the model keeps citing wrong fiscal-year chunks the full bull→bear→synth→verify chain runs again. Capped at 2 in `should_retry`; if it still hallucinates after that we just ship the flagged output.
- SSE stream sometimes drops connection on long analysis runs (~90s+). Browser shows "Connection lost", refresh recovers state. Haven't tracked down whether it's uvicorn keep-alive or the executor.
- BM25 index is a pickle on disk (`data/bm25_index.pkl`). Two parallel ingestion runs will race; one wins, the other's chunks vanish from keyword search until next rebuild.
- `fiscal_year=0` legacy chunks silently bypass the year filter. There's a one-shot `ChunkPipeline.patch_fiscal_years_in_qdrant()` repair function but it needs a manual run.
- Reranker is just an RRF passthrough — CrossEncoder import was killing startup time so the class is stubbed. Plug a real one in once we have a GPU box.
- yfinance returns `info={}` randomly. No caching either, so every analysis re-fetches.
- Anthropic spend isn't bounded per query. Cost runs $0.10–$0.30 per analysis depending on retry count.
- Top-nav links got reduced to "Overview" only — Data Sources and User Guide pages were never built.
