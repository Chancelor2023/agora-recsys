"""Binary relevance metrics on an explicitly ordered, complete candidate list."""
from math import log2


def _validate(labels, k):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if any(type(value) is not int or value not in (0, 1) for value in labels):
        raise ValueError("labels must contain only integer 0 or 1")


def ndcg_at_k(labels, k=10):
    """Return binary NDCG, or None when no candidate is relevant.

    labels are ordered by predicted rank. IDCG uses ALL candidates, not
    just the positive items that happened to be retrieved in the top k.
    """
    labels = list(labels)
    _validate(labels, k)
    positives = sum(labels)
    if not positives:
        return None
    dcg = sum(value / log2(rank + 1)
              for rank, value in enumerate(labels[:k], start=1))
    ideal = sum(1 / log2(rank + 1)
                for rank in range(1, min(k, positives) + 1))
    return dcg / ideal


def recall_at_k(labels, k=20):
    """Fraction of all candidate positives retrieved in the first k ranks."""
    labels = list(labels)
    _validate(labels, k)
    positives = sum(labels)
    return sum(labels[:k]) / positives if positives else None
