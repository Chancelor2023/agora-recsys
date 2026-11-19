"""Load reranking configuration and feeds from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agents import FairnessAgent
from .scruf import ScrufDReranker
from .state import ExposureState


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_reranker(config: dict[str, Any], **overrides: Any) -> ScrufDReranker:
    """Create a reranker from a config dict (see configs/reranking_agents.json)."""
    settings = {
        "lam": config.get("lambda", 0.5),
        "k": config.get("k", 10),
        "allocation": config.get("allocation", "lottery"),
        "window": config.get("window", 50),
        "seed": config.get("seed", 42),
    }
    settings.update({k: v for k, v in overrides.items() if v is not None})
    agents = [FairnessAgent.from_dict(a) for a in config.get("agents", [])]
    if not agents:
        raise ValueError("config must declare at least one fairness agent")
    return ScrufDReranker(agents, state=ExposureState(window=settings.pop("window")), **settings)


def load_feeds(path: str | Path) -> list[dict[str, Any]]:
    """Accept {"feeds": [...]}, a list of feeds, or flat {"predictions": [...]} rows.

    Flat rows (relevance-model output) are grouped by feed_id; each row must
    carry user_id, article_id, relevance_score and the agents' attribute
    (e.g. topic).
    """
    payload = load_json(path)
    if isinstance(payload, dict) and "feeds" in payload:
        return list(payload["feeds"])
    if isinstance(payload, list) and payload and "candidates" in payload[0]:
        return payload
    rows = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected feeds or prediction rows")
    feeds: dict[str, dict[str, Any]] = {}
    for row in rows:
        feed = feeds.setdefault(
            row["feed_id"],
            {"user_id": row["user_id"], "feed_id": row["feed_id"],
             "generated_at": row.get("generated_at", ""), "candidates": []},
        )
        if feed["user_id"] != row["user_id"]:
            raise ValueError(f"feed {row['feed_id']} belongs to multiple users")
        candidate = {k: v for k, v in row.items()
                     if k not in ("user_id", "feed_id", "rank", "generated_at")}
        feed["candidates"].append(candidate)
    return list(feeds.values())
