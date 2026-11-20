"""Position-debiased offline metric (IPS-weighted NDCG).

In IDEA a reader mostly reads what the app displayed at the top, so plain NDCG on
logged reads rewards a ranking for reproducing the old display order. Weighting
each observed read by 1 / P(seen | displayed position) (inverse propensity
scoring, Joachims et al. 2017) corrects this. Propensities are estimated from the
training split only: read rate per displayed position, relative to position 1.
This estimate is confounded with the old ranker's quality, so it is a coarse
correction, reported next to the plain metrics.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


def estimate_propensity(train_feeds, train_labels, positions=26, floor=0.05):
    label = {(r["feed_id"], r["article_id"]): int(r["label"]) for r in train_labels}
    hits, seen = Counter(), Counter()
    for feed in train_feeds:
        for i, article in enumerate(feed["candidates"][:positions]):
            seen[i] += 1
            hits[i] += label[(feed["feed_id"], article)]
    rates = [hits[i] / seen[i] if seen[i] else 0.0 for i in range(positions)]
    top = rates[0] or 1.0
    return [max(r / top, floor) for r in rates]


def ips_ndcg(predictions, labels, feeds, propensity, k=10):
    """Mean IPS-weighted NDCG@k over feeds with at least one read."""
    label = {(r["feed_id"], r["article_id"]): int(r["label"]) for r in labels}
    displayed = {f["feed_id"]: {a: i for i, a in enumerate(f["candidates"])} for f in feeds}
    by_feed = defaultdict(list)
    for row in predictions:
        by_feed[row["feed_id"]].append(row)
    values = []
    for feed_id, rows in by_feed.items():
        rows.sort(key=lambda r: r["rank"])
        last = len(propensity) - 1
        gains = [label[(feed_id, r["article_id"])] / propensity[min(displayed[feed_id][r["article_id"]], last)]
                 for r in rows]
        if not any(gains):
            continue
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
        ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains, reverse=True)[:k]))
        values.append(dcg / ideal)
    return sum(values) / len(values) if values else None
