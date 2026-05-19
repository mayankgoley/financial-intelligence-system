import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import settings
from retrieval.hybrid_retriever import HybridRetriever
from ingestion.market_data import format_market_context
from ingestion.dynamic_ingestor import DynamicIngestor, extract_ticker_from_query
from agents.graph import agent_graph
from agents.state import AgentState

app = FastAPI(title="Financial Intelligence System")
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

retriever = None
ingestor = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = HybridRetriever()
    return retriever

def get_ingestor():
    global ingestor
    if ingestor is None:
        ingestor = DynamicIngestor()
    return ingestor


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/debate", response_class=HTMLResponse)
async def debate_page(request: Request, query: str = "", ticker: str = ""):
    return templates.TemplateResponse("debate.html", {
        "request": request,
        "query": query,
        "ticker": ticker,
    })


async def event_generator(query: str, ticker: str = ""):
    loop = asyncio.get_event_loop()

    if not ticker:
        detected = extract_ticker_from_query(query)
        if detected:
            ticker = detected
            yield f"data: {json.dumps({'agent': 'system', 'status': 'info', 'data': {'message': f'Detected ticker: {ticker}'}})}\n\n"

    if ticker:
        yield f"data: {json.dumps({'agent': 'system', 'status': 'checking', 'data': {'message': f'Checking data for {ticker}...'}})}\n\n"
        await asyncio.sleep(0)

        ready, message = await loop.run_in_executor(
            None, get_ingestor().ensure_company_data, ticker
        )

        if not ready:
            yield f"data: {json.dumps({'agent': 'system', 'status': 'error', 'data': {'message': f'no data for {ticker}: {message}'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield f"data: {json.dumps({'agent': 'system', 'status': 'ready', 'data': {'message': message}})}\n\n"
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'agent': 'retriever', 'status': 'running'})}\n\n"
    await asyncio.sleep(0)

    chunks, timing = await loop.run_in_executor(
        None,
        lambda: get_retriever().retrieve(query, top_k=8, ticker_filter=ticker or None)
    )
    chunk_dicts = [c if isinstance(c, dict) else c.dict() for c in chunks]

    yield f"data: {json.dumps({'agent': 'retriever', 'status': 'complete', 'data': {'chunks': len(chunk_dicts), 'timing': timing}})}\n\n"
    await asyncio.sleep(0)

    if not chunk_dicts:
        no_data_msg = f"No matching docs. Year may not exist for {ticker or 'this company'}."
        yield f"data: {json.dumps({'agent': 'error', 'status': 'failed', 'data': {'message': no_data_msg}})}\n\n"
        yield "data: [DONE]\n\n"
        return

    market_ctx = ""
    if ticker:
        try:
            market_ctx = await loop.run_in_executor(
                None, format_market_context, ticker
            )
        except Exception:
            market_ctx = f"Market data unavailable for {ticker}"

    initial_state = {
        "query": query,
        "market_context": market_ctx,
        "context_chunks": chunk_dicts,
        "ticker": ticker or None,
        "target_year": None,
        "fiscal_year_filter": None,
        "bull_thesis": None,
        "bear_thesis": None,
        "synthesis": None,
        "verified_output": None,
        "hallucination_flags": [],
        "hallucination_rate": 0.0,
        "retry_count": 0,
        "token_usage": {},
        "latency_ms": {},
        "stream_events": [],
    }

    try:
        for agent_name in ["bull", "bear", "synthesis", "verifier"]:
            yield f"data: {json.dumps({'agent': agent_name, 'status': 'waiting'})}\n\n"
        await asyncio.sleep(0)

        result = await loop.run_in_executor(
            None, agent_graph.invoke, initial_state
        )

        for event in result.get("stream_events", []):
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0)

        final_data = result.get("verified_output", {})
        final_data["token_usage"] = result.get("token_usage", {})
        final_data["latency_ms"] = result.get("latency_ms", {})

        yield f"data: {json.dumps({'agent': 'final', 'status': 'complete', 'data': final_data})}\n\n"

    except Exception as e:
        # SSE stream sometimes dies here, browser just sees connection drop
        yield f"data: {json.dumps({'agent': 'error', 'status': 'failed', 'data': {'message': str(e)}})}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/stream")
async def stream_analysis(query: str, ticker: str = ""):
    return StreamingResponse(
        event_generator(query, ticker),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
