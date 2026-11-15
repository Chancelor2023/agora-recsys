"""SCRUF-D-inspired fairness reranker (Aird et al.).

One forward pass for one feed:

1. **Allocation** - every fairness agent computes its fairness from the
   exposure history of the lists already delivered. An agent's weight is
   ``(1 - fairness) * compatibility``. ``lottery`` (SCRUF-D default here)
   draws one agent proportionally to these weights; ``weighted`` keeps all.
2. **Choice** - relevance scores are min-max normalised within the feed, then
   each article receives ``lam * sum(weight_i * 1[article in group_i])``.
   Articles are sorted by this reranking score (ties: relevance, then id).
3. **State update** - the position-weighted top-k exposure of each protected
   group in the delivered list is stored for the next feed.

``lam=0`` is the ablation: the relevance ranking alone.
Relevance scores are preserved unchanged next to the reranking score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .agents import FairnessAgent
from .state import ExposureState, exposure_shares

ALLOCATIONS = ("lottery", "weighted")


@dataclass
class RerankResult:
    user_id: Any
    feed_id: Any
    ranking: list[dict[str, Any]]
    selected_agent: str | None
    agent_weights: dict[str, float]
    agent_fairness: dict[str, float]
    exposure: dict[str, float]

    def predictions(self) -> list[dict[str, Any]]:
        """Rows in the evaluation format (no labels)."""
        return [
            {
                "user_id": self.user_id,
                "feed_id": self.feed_id,
                "article_id": row["article_id"],
                "rank": row["rank"],
                "relevance_score": row["relevance_score"],
                "reranking_score": row["reranking_score"],
            }
            for row in self.ranking
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "feed_id": self.feed_id,
            "selected_agent": self.selected_agent,
            "agent_weights": self.agent_weights,
            "agent_fairness_before": self.agent_fairness,
            "exposure_of_delivered_list": self.exposure,
        }


class ScrufDReranker:
    def __init__(
        self,
        agents: Sequence[FairnessAgent],
        lam: float = 0.5,
        k: int = 10,
        allocation: str = "lottery",
        state: ExposureState | None = None,
        window: int = 50,
        seed: int | None = 42,
        id_key: str = "article_id",
        score_key: str = "relevance_score",
    ) -> None:
        if allocation not in ALLOCATIONS:
            raise ValueError(f"allocation must be one of {ALLOCATIONS}")
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        if lam < 0:
            raise ValueError("lam must be >= 0")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.agents = list(agents)
        self.lam = lam
        self.k = k
        self.allocation = allocation
        self.state = state if state is not None else ExposureState(window=window)
        self.rng = random.Random(seed)
        self.id_key = id_key
        self.score_key = score_key

    # ------------------------------------------------------------------ #
    # 1. Allocation
    # ------------------------------------------------------------------ #
    def allocate(
        self, compatibility: Mapping[str, float] | None = None
    ) -> tuple[dict[str, float], dict[str, float], str | None]:
        """Return (weights, fairness, selected_agent)."""
        compatibility = compatibility or {}
        fairness = {
            a.name: a.fairness(self.state.observed_share(a.name)) for a in self.agents
        }
        raw = {
            a.name: max(0.0, (1.0 - fairness[a.name]) * float(compatibility.get(a.name, 1.0)))
            for a in self.agents
        }
        total = sum(raw.values())
        if total <= 0:
            return {name: 0.0 for name in raw}, fairness, None

        if self.allocation == "weighted":
            weights = {name: w / total for name, w in raw.items()}
            selected = max(weights, key=lambda n: (weights[n], n))
            return weights, fairness, selected

        # lottery: one agent wins, with probability proportional to its weight
        draw = self.rng.random() * total
        cumulative = 0.0
        selected = None
        for name, w in raw.items():
            cumulative += w
            if w > 0 and draw < cumulative:
                selected = name
                break
        if selected is None:  # floating-point edge case
            selected = max(raw, key=lambda n: (raw[n], n))
        return {n: (1.0 if n == selected else 0.0) for n in raw}, fairness, selected

    # ------------------------------------------------------------------ #
    # 2. Choice (rescoring)
    # ------------------------------------------------------------------ #
    def choose(
        self,
        candidates: Sequence[Mapping[str, Any]],
        weights: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        self._validate(candidates)
        scores = [float(c[self.score_key]) for c in candidates]
        lo, hi = min(scores), max(scores)
        span = hi - lo

        enriched = []
        for c, s in zip(candidates, scores):
            base = (s - lo) / span if span > 0 else 1.0
            bonus = sum(weights.get(a.name, 0.0) for a in self.agents if a.is_protected(c))
            enriched.append((base + self.lam * bonus, s, str(c[self.id_key]), c))

        enriched.sort(key=lambda t: (-t[0], -t[1], t[2]))
        ranking = []
        for rank, (reranking_score, _, _, c) in enumerate(enriched, start=1):
            item = dict(c)  # keep metadata (topic, outlet...) for exposure + API
            item["rank"] = rank
            item["reranking_score"] = round(reranking_score, 6)
            ranking.append(item)
        return ranking

    # ------------------------------------------------------------------ #
    # Full forward pass for one feed
    # ------------------------------------------------------------------ #
    def rerank(
        self,
        feed: Mapping[str, Any],
        compatibility: Mapping[str, float] | None = None,
        update_state: bool = True,
    ) -> RerankResult:
        """Rerank one feed: {"user_id", "feed_id", "candidates": [...]}."""
        weights, fairness, selected = self.allocate(compatibility)
        ranking = self.choose(feed.get("candidates") or [], weights)
        shares = exposure_shares(ranking, self.agents, self.k)
        if update_state:
            self.state.update(shares)
        return RerankResult(
            user_id=feed.get("user_id"),
            feed_id=feed.get("feed_id"),
            ranking=ranking,
            selected_agent=selected,
            agent_weights={n: round(w, 6) for n, w in weights.items()},
            agent_fairness={n: round(f, 6) for n, f in fairness.items()},
            exposure={n: round(s, 6) for n, s in shares.items()},
        )

    def rerank_feeds(
        self, feeds: Sequence[Mapping[str, Any]]
    ) -> list[RerankResult]:
        """Rerank feeds in chronological order (``generated_at`` if present)."""
        ordered = sorted(
            enumerate(feeds),
            key=lambda t: (str(t[1].get("generated_at", "")), t[0]),
        )
        return [self.rerank(feed) for _, feed in ordered]

    # ------------------------------------------------------------------ #
    def _validate(self, candidates: Sequence[Mapping[str, Any]]) -> None:
        if not candidates:
            raise ValueError("feed has no candidates")
        ids = []
        for c in candidates:
            if self.id_key not in c or self.score_key not in c:
                raise KeyError(f"each candidate needs '{self.id_key}' and '{self.score_key}'")
            if "label" in c:
                raise ValueError("labels must never be passed to the reranker")
            ids.append(c[self.id_key])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate article ids in feed")
