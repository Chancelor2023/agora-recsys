"""Fairness agents for the SCRUF-D-inspired reranker.

Each agent protects one group of articles (e.g. public-affairs news, an
under-exposed source). It measures how fairly that group has been exposed in
recent recommendation lists and expresses a preference over candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FairnessAgent:
    """One fairness concern, in the sense of SCRUF-D (Aird et al.).

    name:             unique identifier of the agent.
    attribute:        candidate field used to decide group membership
                      (e.g. "category", "source").
    protected_values: values of ``attribute`` that belong to the protected group.
    target_share:     desired share of position-weighted exposure for the group
                      over the recent window, in (0, 1].
    """

    name: str
    attribute: str
    protected_values: frozenset = field(default_factory=frozenset)
    target_share: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.target_share <= 1.0:
            raise ValueError(f"{self.name}: target_share must be in (0, 1]")
        # Accept any iterable for convenience, store a frozenset.
        object.__setattr__(self, "protected_values", frozenset(self.protected_values))

    def is_protected(self, candidate: Mapping[str, Any]) -> bool:
        """True if the candidate article belongs to this agent's group."""
        value = candidate.get(self.attribute)
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(v in self.protected_values for v in value)
        return value in self.protected_values

    def fairness(self, observed_share: float | None) -> float:
        """Fairness score in [0, 1]; 1 means the target exposure is reached.

        ``None`` (no history yet) is treated as perfectly fair, so the reranker
        behaves like the base model on the very first feed.
        """
        if observed_share is None:
            return 1.0
        return min(1.0, observed_share / self.target_share)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FairnessAgent":
        return cls(
            name=data["name"],
            attribute=data["attribute"],
            protected_values=frozenset(data.get("protected_values", [])),
            target_share=float(data.get("target_share", 0.2)),
        )
