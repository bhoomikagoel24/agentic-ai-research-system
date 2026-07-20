"""
ARIA FastAPI Backend
====================
Exposes the LangGraph pipeline as a REST API.

Routes:
  GET  /health          — liveness check (used by Docker HEALTHCHECK)
  GET  /                — basic info
  POST /research        — run pipeline, return full result (blocking, ~8-15 min)
  POST /research/stream — run pipeline, stream per-node updates as SSE

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import time
import hashlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from research_agent_system.config.config import Config
from research_agent_system.workflows.langgraph_workflow import build_graph
from research_agent_system.workflows.research_pipeline import evaluate_pipeline
from research_agent_system.schemas import PipelineState


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300,
                       description="The research topic to investigate")
    top_k: int = Field(default=12, ge=3, le=40,
                       description="Number of papers to keep for synthesis")
    dev_mode: bool = Field(default=False,
                            description="Fast mode: 2 papers, 1 retry (for testing)")


class PaperOut(BaseModel):
    title: str
    abstract: str
    url: str | None
    year: int | None
    source: str
    relevance_score: float | None


class MetricsOut(BaseModel):
    papers_retrieved: int
    summaries_generated: int
    avg_relevance: float
    avg_quality: float
    high_confidence: int
    status: str


class ResearchResponse(BaseModel):
    topic: str
    final_report: str
    papers: list[PaperOut]
    metrics: MetricsOut
    critique_retries: int
    duration_seconds: float


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing to pre-load right now.
    # When you add a persistent DB connection or a pre-loaded
    # embedding model, do it here instead of at import time.
    yield


app = FastAPI(
    title="ARIA — Autonomous Research Intelligence API",
    description=(
        "Multi-agent research synthesis pipeline. "
        "Give it a topic, get back a cited, self-critiqued research report."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _build_config(req: ResearchRequest) -> Config:
    return Config(
        dev_mode=req.dev_mode,
        topk=req.top_k,
    )


def _thread_id(topic: str) -> str:
    """Deterministic checkpoint key per topic — same topic resumes last run."""
    return hashlib.md5(topic.encode("utf-8")).hexdigest()


def _to_response(topic: str, accumulated: dict, duration: float) -> ResearchResponse:
    """Convert the accumulated LangGraph state dict into the API response shape."""
    papers_raw = accumulated.get("papers", [])
    papers_out = [
        PaperOut(
            title=p.get("title", ""),
            abstract=p.get("abstract", "")[:400],  # truncate for API response size
            url=p.get("url"),
            year=p.get("year"),
            source=p.get("source", "unknown"),
            relevance_score=p.get("relevance_score"),
        )
        for p in papers_raw
    ]

    # Re-use the same metrics logic the Streamlit UI uses
    # by constructing a temporary PipelineState
    tmp_state = PipelineState(
        topic=topic,
        plan=accumulated.get("plan"),
        papers=papers_raw,
        summaries=accumulated.get("summaries", []),
        synthesis=accumulated.get("synthesis"),
        critique=accumulated.get("critique", {}),
        final_report=accumulated.get("final_report", ""),
    )
    raw_metrics = evaluate_pipeline(tmp_state)
    # evaluate_pipeline returns a list of (label, value, icon, color) tuples
    metrics_dict = {label: value for label, value, *_ in raw_metrics}

    metrics_out = MetricsOut(
        papers_retrieved=int(metrics_dict.get("Papers Retrieved", len(papers_raw))),
        summaries_generated=int(metrics_dict.get("Summaries", len(accumulated.get("summaries", [])))),
        avg_relevance=float(metrics_dict.get("Avg Relevance", 0.0)),
        avg_quality=float(str(metrics_dict.get("Avg Quality", "0")).split("/")[0]),
        high_confidence=int(metrics_dict.get("High Confidence", 0)),
        status=str(metrics_dict.get("Status", "complete")),
    )

    return ResearchResponse(
        topic=topic,
        final_report=accumulated.get("final_report", ""),
        papers=papers_out,
        metrics=metrics_out,
        critique_retries=accumulated.get("critique_retries", 0),
        duration_seconds=round(duration, 1),
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "ARIA",
        "version": "2.0.0",
        "description": "Autonomous Research Intelligence — multi-agent LangGraph pipeline",
        "routes": {
            "GET  /health": "liveness check",
            "POST /research": "blocking full pipeline run (~8-15 min)",
            "POST /research/stream": "streaming SSE — get updates as each node finishes",
        }
    }


@app.get("/health")
def health():
    """
    Docker HEALTHCHECK and load-balancer probe target.
    Returns 200 as long as the process is alive.
    Does NOT check Gemini/Groq connectivity — a slow/rate-limited LLM
    shouldn't make the container look unhealthy to the scheduler.
    """
    return {"status": "ok", "service": "aria-api"}


@app.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    """
    Blocking research pipeline.

    Runs the full LangGraph graph synchronously and returns the complete
    report when done. Suitable for backend-to-backend calls where the
    caller can hold an open connection for 8-15 minutes.

    For a user-facing frontend, use POST /research/stream instead.
    """
    cfg = _build_config(req)
    graph = build_graph(cfg)
    run_config = {"configurable": {"thread_id": _thread_id(req.topic)}}

    start = time.time()
    accumulated: dict = {"topic": req.topic}

    try:
        # run_until_complete lets us call the sync graph.stream()
        # from this async route without blocking the event loop
        loop = asyncio.get_event_loop()
        def _run():
            for update in graph.stream(
                {"topic": req.topic},
                config=run_config,
                stream_mode="updates"
            ):
                for _, partial in update.items():
                    accumulated.update(partial)
        await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    if not accumulated.get("final_report"):
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed but produced no report. Check logs."
        )

    return _to_response(req.topic, accumulated, time.time() - start)


@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    """
    Streaming research pipeline via Server-Sent Events (SSE).

    Each node completion is pushed to the client immediately as it
    finishes, so the frontend can show live progress instead of waiting
    for the full result.

    Event format (JSON lines):
      {"event": "node_done", "node": "planner", "data": {...partial_state}}
      {"event": "node_done", "node": "research", "data": {...}}
      ...
      {"event": "complete", "data": {...full_response}}
      {"event": "error",    "data": {"message": "..."}}

    JavaScript usage:
      const es = new EventSource('/research/stream', {method: 'POST', body: JSON.stringify({topic: "..."}), ...})
      (or use fetch + ReadableStream for POST SSE)
    """
    cfg = _build_config(req)
    graph = build_graph(cfg)
    run_config = {"configurable": {"thread_id": _thread_id(req.topic)}}

    async def event_generator() -> AsyncGenerator[str, None]:
        accumulated: dict = {"topic": req.topic}
        start = time.time()

        try:
            loop = asyncio.get_event_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _run_graph():
                try:
                    for update in graph.stream(
                        {"topic": req.topic},
                        config=run_config,
                        stream_mode="updates"
                    ):
                        queue.put_nowait(("update", update))
                    queue.put_nowait(("done", None))
                except Exception as e:
                    queue.put_nowait(("error", str(e)))

            # Run the blocking graph in a thread, drain events in async
            loop.run_in_executor(None, _run_graph)

            while True:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=900)

                if kind == "error":
                    yield f"data: {json.dumps({'event': 'error', 'data': {'message': payload}})}\n\n"
                    break

                if kind == "done":
                    response = _to_response(req.topic, accumulated, time.time() - start)
                    yield f"data: {json.dumps({'event': 'complete', 'data': response.model_dump()})}\n\n"
                    break

                # kind == "update"
                for node_name, partial in payload.items():
                    accumulated.update(partial)
                    # Send partial state — omit large fields (papers/summaries)
                    # for bandwidth; send only what the frontend needs to show progress
                    safe_partial = {
                        k: v for k, v in partial.items()
                        if k not in ("papers", "summaries", "synthesis")
                    }
                    event = {
                        "event": "node_done",
                        "node": node_name,
                        "data": safe_partial,
                        "critique_retries": accumulated.get("critique_retries", 0),
                    }
                    yield f"data: {json.dumps(event, default=str)}\n\n"

        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Pipeline timed out after 15 minutes'}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # tells Nginx not to buffer SSE
        },
    )
