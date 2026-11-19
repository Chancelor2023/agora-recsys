"""FastAPI entry point for the Agora recommendation service.

Real mode (default when data/processed and artifacts/models exist): the relevance
model scores the feed's candidates, then the SCRUF-D reranker reorders them.
Mock mode (no artifacts, or AGORA_MOCK=1): synthetic items from
examples/feed_input.json, as in the first API version.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from agora.data.storage import InteractionStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FEED_FILE = PROJECT_ROOT / "examples" / "feed_input.json"

MODEL_VERSION = "mock-v1"


class RecommendationItem(BaseModel):
    """One recommended article."""

    article_id: str
    rank: int
    title: str
    lead: str
    outlet: str
    published_at: datetime
    topic: Optional[str] = None
    url: Optional[str] = None
    image: Optional[str] = None
    relevance_score: Optional[float] = None
    reranking_score: Optional[float] = None


class RecommendationResponse(BaseModel):
    """Recommendation response returned by the API."""

    user_id: str
    feed_id: str
    model_version: str
    is_mock: bool
    items: list[RecommendationItem]
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Basic API health response."""

    status: str


class InteractionEvent(BaseModel):
    """One application event sent by the frontend."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(min_length=1)
    feed_id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    event_type: Literal["display", "click"]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    displayed_rank: int = Field(ge=1)
    model_version: str = Field(min_length=1)


def load_mock_items() -> list[RecommendationItem]:
    """Load the synthetic candidates from the shared example file."""
    with FEED_FILE.open(encoding="utf-8") as file:
        feed = json.load(file)

    if not feed.get("is_synthetic", False):
        raise ValueError("The example feed must be explicitly marked synthetic.")

    return [
        RecommendationItem(
            article_id=candidate["article_id"],
            rank=rank,
            title=candidate["title"],
            lead=candidate["lead"],
            outlet=candidate["outlet"],
            published_at=candidate["published_at"],
            topic=candidate.get("topic"),
        )
        for rank, candidate in enumerate(feed["candidates"], start=1)
    ]


def _default_service():
    if os.environ.get("AGORA_MOCK") == "1":
        return None
    from agora.api.service import RecommenderService, ServicePaths

    paths = ServicePaths.from_env(PROJECT_ROOT)
    return RecommenderService.from_paths(paths) if paths.ready() else None


def _default_store():
    value = os.environ.get("AGORA_DB", "artifacts/agora.db")
    path = Path(value)
    return InteractionStore(path if path.is_absolute() or value == ":memory:" else PROJECT_ROOT / path)


def create_app(service=None, store=None, mock: bool = False) -> FastAPI:
    """Build the API. ``service=None`` and ``mock=False`` load the real service if available."""
    if not mock and service is None:
        service = _default_service()
    if mock:
        service = None
    store = store if store is not None else _default_store()

    app = FastAPI(
        title="Agora Recommendation API",
        description="News recommendations: relevance model + SCRUF-D fairness reranking.",
        version="0.2.0",
    )
    app.state.service = service
    app.state.store = store

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Check that the API is running."""
        return HealthResponse(status="ok")

    @app.get("/mode")
    def mode() -> dict:
        """Say whether real predictions are served."""
        if service is None:
            return {"is_mock": True, "model_version": MODEL_VERSION}
        return {"is_mock": False, "model_version": service.model_version,
                "users": len(service.users()), "feeds": len(service.feeds)}

    @app.get("/users")
    def users() -> list[str]:
        """Readers with servable feeds (validation and test periods)."""
        return [] if service is None else service.users()

    @app.get("/users/{user_id}/feeds")
    def user_feeds(user_id: str) -> list[dict]:
        if service is None:
            return []
        try:
            return service.feeds_for(user_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/recommend", response_model=RecommendationResponse)
    def recommend(user_id: str, feed_id: str) -> RecommendationResponse:
        """Return the reranked feed (or synthetic items in mock mode)."""
        start = time.perf_counter()
        if service is None:
            return RecommendationResponse(user_id=user_id, feed_id=feed_id,
                                          model_version=MODEL_VERSION, is_mock=True,
                                          items=load_mock_items())
        try:
            items = service.recommend(user_id, feed_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RecommendationResponse(
            user_id=user_id, feed_id=feed_id, model_version=service.model_version,
            is_mock=False, items=[RecommendationItem(**i) for i in items],
            latency_ms=round((time.perf_counter() - start) * 1000, 1))

    @app.get("/live/status")
    def live_status() -> dict:
        """Whether today's articles are loaded (run python -m agora.data.live)."""
        if service is None:
            return {"available": False, "reason": "mock mode"}
        return service.live_status()

    @app.get("/recommend_live", response_model=RecommendationResponse)
    def recommend_live(user_id: str, limit: int = Query(20, ge=1, le=200)) -> RecommendationResponse:
        """Recommend today's articles (RSS) to a reader, with the same model and reranker."""
        start = time.perf_counter()
        if service is None:
            raise HTTPException(status_code=503, detail="Live recommendations need the real service")
        try:
            items = service.recommend_live(user_id, limit)
        except LookupError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return RecommendationResponse(
            user_id=user_id, feed_id="live", model_version=service.model_version, is_mock=False,
            items=[RecommendationItem(**i) for i in items],
            latency_ms=round((time.perf_counter() - start) * 1000, 1))

    @app.post("/log_interaction")
    def log_interaction(event: InteractionEvent) -> dict:
        """Persist one display or click event in SQLite."""
        row = event.model_dump()
        row["timestamp"] = event.timestamp.isoformat()
        stored = store.insert(row)
        return {"stored": stored, "event_id": event.event_id, "total_events": store.count()}

    @app.get("/interactions")
    def interactions(user_id: Optional[str] = None, limit: int = Query(20, ge=1, le=500)) -> list[dict]:
        """Most recent stored events (evidence for the report)."""
        return store.recent(user_id, limit)

    return app


app = create_app()
