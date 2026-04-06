"""
FastAPI REST API for the Fake News Detection system.

Endpoints:
  POST /detect          — full pipeline
  POST /detect/batch    — batch inference
  POST /classify        — BERT only
  POST /factcheck       — fact-check only
  POST /graph           — graph analysis only
  GET  /health          — liveness check

Run with:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 2
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=2000)
    body: str = Field("", max_length=20000)
    language: str = Field("en", pattern=r"^[a-z]{2}$")
    propagation_graph: Optional[dict] = None

class BatchRequest(BaseModel):
    articles: list[ArticleRequest] = Field(..., max_items=50)

class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000)
    language: str = "en"

class GraphRequest(BaseModel):
    root_id: str
    nodes: list[dict]
    edges: list[dict]

_pipeline = None
_classifier = None
_fact_checker = None
_graph_analyzer = None


def get_pipeline():
    global _pipeline, _classifier, _fact_checker, _graph_analyzer
    if _pipeline is None:
        from pipeline.orchestrator import FakeNewsDetectionPipeline
        from factcheck.client import FactChecker
        from graph.propagation import PropagationAnalyzer

        google_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
        claimbuster_key = os.getenv("CLAIMBUSTER_API_KEY", "")
        model_path = os.getenv("MODEL_CHECKPOINT", "")

        _fact_checker = FactChecker(
            google_api_key=google_key,
            claimbuster_api_key=claimbuster_key,
        )
        _graph_analyzer = PropagationAnalyzer()

        if model_path:
            model_file = os.path.join(model_path, "config.json")
            if os.path.exists(model_file):
                from classifier.model import FakeNewsClassifier
                _classifier = FakeNewsClassifier.load(model_path)
                logger.info(f"Loaded BERT model from {model_path}")
            else:
                logger.warning(f"Model files not found at {model_path} — BERT disabled")
                _classifier = None
        else:
            logger.warning("No MODEL_CHECKPOINT set — BERT component disabled")
            _classifier = None

        _pipeline = FakeNewsDetectionPipeline(
            classifier=_classifier,
            fact_checker=_fact_checker,
            graph_analyzer=_graph_analyzer,
        )
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_pipeline()
    except Exception as e:
        logger.error(f"Pipeline init failed: {e}")
    yield

app = FastAPI(
    title="Fake News Detector API",
    description="Multilingual fake news detection: BERT + Fact-Check + Graph Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "bert_loaded": _classifier is not None,
        "factcheck_enabled": bool(os.getenv("GOOGLE_FACTCHECK_API_KEY")),
        "graph_enabled": True,
    }


@app.post("/detect")
async def detect(req: ArticleRequest):
    """Full pipeline detection on a single article."""
    pipeline = get_pipeline()
    try:
        result = pipeline.detect(
            title=req.title,
            body=req.body,
            propagation_graph=req.propagation_graph,
            language=req.language,
        )
        return result.__dict__
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(req: BatchRequest):
    """Batch detection — up to 50 articles."""
    pipeline = get_pipeline()
    try:
        articles = [
            {
                "title": a.title,
                "body": a.body,
                "language": a.language,
                "graph": a.propagation_graph,
            }
            for a in req.articles
        ]
        results = pipeline.detect_batch(articles)
        return [r.__dict__ for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify")
async def classify_only(req: ArticleRequest):
    """BERT-only classification, no fact-check or graph."""
    pipeline = get_pipeline()
    if not pipeline.classifier:
        raise HTTPException(status_code=503, detail="BERT model not loaded")
    result = pipeline._run_bert(req.title, req.body)
    return result


@app.post("/factcheck")
async def factcheck_only(req: ClaimRequest):
    """Fact-check a single claim via Google Fact Check API."""
    pipeline = get_pipeline()
    if not pipeline.fact_checker:
        raise HTTPException(status_code=503, detail="Fact-checker not configured")
    result = pipeline.fact_checker.check(req.claim, req.language)
    return result


@app.post("/graph")
async def graph_only(req: GraphRequest):
    """Analyze a propagation graph."""
    from graph.propagation import GraphBuilder, PropagationAnalyzer
    from dataclasses import asdict
    try:
        g = GraphBuilder.from_dict({
            "root_id": req.root_id,
            "nodes": req.nodes,
            "edges": req.edges,
        })
        analyzer = PropagationAnalyzer()
        features = analyzer.analyze(g)
        return asdict(features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))