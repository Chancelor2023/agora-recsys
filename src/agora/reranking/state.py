"""Exposure history shared by all fairness agents.

SCRUF-D is *dynamic*: the decision for feed t depends on how fair feeds
t-W..t-1 were. This module stores, for each delivered list, the share of
position-weighted exposure that went to each agent's protected group.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .agents import FairnessAgent


def position_weight(rank: int) -> float:
    """DCG-style weight of a 1-based rank (top positions count more)."""
    return 1.0 / math.log2(rank + 1)


def exposure_shares(
    ranked: Sequence[Mapping[str, Any]],
    agents: Iterable[FairnessAgent],
    k: int,
) -> dict[str, float]:
    """Share of top-k position-weighted exposure given to each agent's group."""
    top = list(ranked)[:k]
    total = sum(position_weight(r) for r in range(1, len(top) + 1))
    shares: dict[str, float] = {}
    for agent in agents:
        protected = sum(
            position_weight(r)
            for r, cand in enumerate(top, start=1)
            if agent.is_protected(cand)
        )
        shares[agent.name] = protected / total if total > 0 else 0.0
    return shares


class ExposureState:
    """Sliding window of exposure shares over the last ``window`` lists."""

    def __init__(self, window: int = 50) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.history: deque[dict[str, float]] = deque(maxlen=window)
        self.n_updates = 0

    def observed_share(self, agent_name: str) -> float | None:
        """Mean exposure share of an agent's group over the window."""
        values = [h[agent_name] for h in self.history if agent_name in h]
        if not values:
            return None
        return sum(values) / len(values)

    def update(self, shares: Mapping[str, float]) -> None:
        """Record the exposure of one delivered list."""
        self.history.append(dict(shares))
        self.n_updates += 1

    # --- persistence (so the state survives between API calls / runs) ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "n_updates": self.n_updates,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExposureState":
        state = cls(window=int(data.get("window", 50)))
        for h in data.get("history", []):
            state.history.append(dict(h))
        state.n_updates = int(data.get("n_updates", len(state.history)))
        return state

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ExposureState":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
