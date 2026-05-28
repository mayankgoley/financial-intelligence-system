# financial intelligence system

Multi agent bull versus bear debate over SEC filings with a per claim hallucination check and a live market data sidecar.

## architecture and stack

* ingestion: pulls 10K and 10Q forms from SEC EDGAR. Chunks via langchain splitter. Embeds with OpenAI large models and indexes into Qdrant plus a pickled BM25.
* retrieval: hybrid dense and BM25 with RRF fusion, plus a post rerank fiscal year filter.
* agents: LangGraph state machine moving from bull to bear to synthesis to verifier. Includes a conditional retry edge that pulls chunks again under a strict year filter if the verifier flags hallucinations.
* api: FastAPI with Server Sent Events, Jinja2 templates, and Tailwind via CDN.
* core: Python 3.12, FastAPI, LangGraph, LangChain
* models: OpenAI for embeddings, Anthropic Claude sonnet for the bull and synthesis and verify agents, haiku for the bear agent
* infra: Qdrant for vectors, BM25 for keywords, Postgres for run logs, Redis for semantic cache, yfinance for live market data, docker compose for local setup.

## local setup

Copy the env example file to a new env file.
Fill in your OPENAI API KEY and ANTHROPIC API KEY.

Start your docker containers for postgres, redis, qdrant, kafka, and zookeeper.
Install your requirements via pip.
Run the database python module once to create your tables.
Start the uvicorn server on port 8000 with reload enabled.

Open localhost:8000 and submit a query. The first hit per ticker auto ingests filings from EDGAR and takes about 30 to 60 seconds. Subsequent queries hit Qdrant directly.

Postgres is mapped to host port 5433 because port 5432 was taken on my dev box. The env file matches this.

## known issues and todo

* SEC EDGAR rate limit is annoying. I just put a flat sleep between requests in the rate limit function. It works fine for serial ingestion but throws forbidden errors if you parallelize. I should swap in a token bucket eventually.
* LangGraph retry edge sometimes loops too many times. When the model keeps citing wrong fiscal year chunks the full agent chain runs again. I capped it at 2 in the retry function. If it still hallucinates after that we just ship the flagged output.
* The server stream sometimes drops connection on long analysis runs past 90 seconds. The browser shows connection lost and a refresh recovers state. I have not tracked down whether it is the uvicorn keep alive or the executor.
* BM25 index is a pickle on disk. Two parallel ingestion runs will race. One wins and the other chunks vanish from keyword search until the next rebuild.
* Legacy chunks with fiscal year zero silently bypass the year filter. There is a one shot repair function but it needs a manual run.
* Reranker is just an RRF passthrough. The CrossEncoder import was killing startup time so the class is stubbed. I will plug a real one in once we have a GPU box.
* yfinance returns empty info randomly. There is no caching either so every analysis fetches the data again.
* Anthropic spend is not bounded per query. Cost runs roughly 10 to 30 cents per analysis depending on the retry count.
* Top nav links got reduced to Overview only. The Data Sources and User Guide pages were never built.
