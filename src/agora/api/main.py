
"""FastAPI entry point for the Agora recommendation service."""

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


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


class RecommendationResponse(BaseModel):
    """Recommendation response returned by the API."""

    user_id: str
    feed_id: str
    model_version: str
    is_mock: bool
    items: list[RecommendationItem]


class HealthResponse(BaseModel):
    """Basic API health response."""

    status: str


def load_mock_items() -> list[RecommendationItem]:
    """Load the synthetic candidates from the shared example file."""
    with FEED_FILE.open(encoding="utf-8") as file:
        feed = json.load(file)

    if not feed.get("is_synthetic", False):
        raise ValueError("The example feed must be explicitly marked synthetic.")

    items = []

    for rank, candidate in enumerate(feed["candidates"], start=1):
        items.append(
            RecommendationItem(
                article_id=candidate["article_id"],
                rank=rank,
                title=candidate["title"],
                lead=candidate["lead"],
                outlet=candidate["outlet"],
                published_at=candidate["published_at"],
            )
        )

    return items


app = FastAPI(
    title="Agora Recommendation API",
    description="Mock recommendation API for the Agora project.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check that the API is running."""
    return HealthResponse(status="ok")


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(user_id: str, feed_id: str) -> RecommendationResponse:
    """Return synthetic recommendations for the requested user and feed."""
    return RecommendationResponse(
        user_id=user_id,
        feed_id=feed_id,
        model_version=MODEL_VERSION,
        is_mock=True,
        items=load_mock_items(),
    )
