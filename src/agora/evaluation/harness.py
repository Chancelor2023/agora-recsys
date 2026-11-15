"""Validate predictions against separate labels, then macro-average by feed."""
import math
from .metrics import ndcg_at_k, recall_at_k


def _identifier(row, field):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _group(rows, predictions):
    groups = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each row must be an object")
        feed = _identifier(row, "feed_id")
        article = _identifier(row, "article_id")
        if (feed, article) in seen:
            raise ValueError(f"duplicate article {article} in feed {feed}")
        seen.add((feed, article))
        if predictions:
            _identifier(row, "user_id")
            if type(row.get("rank")) is not int or row["rank"] < 1:
                raise ValueError("rank must be a positive integer")
            score = row.get("relevance_score")
            if (type(score) not in (int, float) or not math.isfinite(score)):
                raise ValueError("relevance_score must be finite")
            if "label" in row:
                raise ValueError("labels must be separate from predictions")
        elif type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        groups.setdefault(feed, []).append(row)
    if not groups:
        raise ValueError("input must contain at least one feed")
    return groups


def evaluate(predictions, labels, ndcg_k=10, recall_k=20):
    """Evaluate full candidate rankings; fail on missing or extra labels.

    Every feed has equal weight, even if users have different feed counts.
    Ranks are authoritative; equal relevance scores need an explicit upstream
    tie-break. No-positive feeds are counted and excluded from both means.
    """
    # Validate cutoffs even when all feeds have no positives.
    ndcg_at_k([], ndcg_k)
    recall_at_k([], recall_k)
    predicted = _group(predictions, True)
    truth = _group(labels, False)
    if predicted.keys() != truth.keys():
        raise ValueError("prediction and label feed sets must match")
    per_feed = []
    for feed in sorted(predicted):
        rows = sorted(predicted[feed], key=lambda row: row["rank"])
        if len({row["user_id"] for row in rows}) != 1:
            raise ValueError(f"feed {feed} belongs to multiple users")
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"feed {feed} ranks must be contiguous from 1")
        by_article = {row["article_id"]: row["label"] for row in truth[feed]}
        if {row["article_id"] for row in rows} != by_article.keys():
            raise ValueError(f"feed {feed} candidate sets must match exactly")
        ranked_labels = [by_article[row["article_id"]] for row in rows]
        positives = sum(ranked_labels)
        per_feed.append({
            "user_id": rows[0]["user_id"], "feed_id": feed,
            "candidate_count": len(rows), "positive_count": positives,
            "excluded": not bool(positives),
            f"NDCG@{ndcg_k}": ndcg_at_k(ranked_labels, ndcg_k),
            f"Recall@{recall_k}": recall_at_k(ranked_labels, recall_k),
        })
    included = [row for row in per_feed if not row["excluded"]]
    metrics = {}
    for key in (f"NDCG@{ndcg_k}", f"Recall@{recall_k}"):
        metrics[key] = (sum(row[key] for row in included) / len(included)
                        if included else None)
    return {
        "aggregation": "macro_by_feed",
        "zero_positive_policy": "exclude_and_report_count",
        "total_feeds": len(per_feed), "evaluated_feeds": len(included),
        "excluded_zero_positive_feeds": len(per_feed) - len(included),
        "metrics": metrics, "per_feed": per_feed,
    }
