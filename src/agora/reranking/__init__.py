"""SCRUF-D-inspired agents, allocation, rescoring and exposure state."""

from .agents import FairnessAgent
from .io import build_reranker, load_feeds
from .scruf import RerankResult, ScrufDReranker
from .state import ExposureState, exposure_shares

__all__ = [
    "FairnessAgent",
    "ExposureState",
    "RerankResult",
    "ScrufDReranker",
    "build_reranker",
    "exposure_shares",
    "load_feeds",
]
