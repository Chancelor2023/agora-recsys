"""Paired bootstrap confidence interval for the difference between two rankings.

Both rankings are scored by the same harness (``evaluate``) on the same labels,
so they share exactly the same evaluated feeds (feeds with at least one positive).
The statistic is the difference of the macro-by-feed means, A minus B, as in the
reported tables. Resampling unit:

- ``user`` (default): readers are drawn with replacement and keep all their
  evaluated feeds, because feeds of the same reader are not independent;
- ``feed``: feeds are drawn with replacement (assumes independent feeds).

The interval is the percentile interval of the resampled differences. Runs are
deterministic for a given seed.

    python -m agora.evaluation.bootstrap --labels data/processed/validation_labels.json \\
        --a predictions_a.json --name-a "Logistic regression" \\
        --b predictions_b.json --name-b "Neural" --output reports/week3/bootstrap_validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .__main__ import _load
from .harness import evaluate


def _evaluated_rows(result):
    return {row["feed_id"]: row for row in result["per_feed"] if not row["excluded"]}


def paired_bootstrap(result_a, result_b, metric="NDCG@10", unit="user",
                     resamples=2000, seed=42, alpha=0.05):
    """Return the observed difference A - B and its bootstrap interval.

    ``result_a`` and ``result_b`` are outputs of ``evaluate`` on the same labels.
    """
    if unit not in ("user", "feed"):
        raise ValueError("unit must be 'user' or 'feed'")
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    all_a = {row["feed_id"] for row in result_a["per_feed"]}
    all_b = {row["feed_id"] for row in result_b["per_feed"]}
    rows_a, rows_b = _evaluated_rows(result_a), _evaluated_rows(result_b)
    if all_a != all_b or rows_a.keys() != rows_b.keys():
        raise ValueError("both rankings must be evaluated on the same feeds and labels")
    if not rows_a:
        raise ValueError("no evaluated feed (every feed has zero positives)")
    feeds = sorted(rows_a)
    for feed in feeds:
        if rows_a[feed]["user_id"] != rows_b[feed]["user_id"]:
            raise ValueError(f"feed {feed} has different readers in A and B")
        if metric not in rows_a[feed] or metric not in rows_b[feed]:
            raise ValueError(f"metric {metric} missing from the evaluation results")

    diff = np.array([rows_a[f][metric] - rows_b[f][metric] for f in feeds], dtype=float)
    keys = [rows_a[f]["user_id"] if unit == "user" else f for f in feeds]
    units = sorted(set(keys))
    index = {key: i for i, key in enumerate(units)}
    sums = np.zeros(len(units))
    counts = np.zeros(len(units))
    for key, value in zip(keys, diff):
        sums[index[key]] += value
        counts[index[key]] += 1

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(units), size=(resamples, len(units)))
    # Macro-by-feed mean of each resample: total difference / number of feeds drawn.
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    mean_a = float(np.mean([rows_a[f][metric] for f in feeds]))
    mean_b = float(np.mean([rows_b[f][metric] for f in feeds]))
    return {
        "metric": metric, "unit": unit, "resamples": resamples, "seed": seed,
        "confidence": 1 - alpha, "evaluated_feeds": len(feeds), "units": len(units),
        "mean_a": mean_a, "mean_b": mean_b, "difference_a_minus_b": float(diff.mean()),
        "ci_low": float(low), "ci_high": float(high),
        "share_of_resamples_above_zero": float((stats > 0).mean()),
    }


def compare(predictions_a, predictions_b, labels, metrics=("NDCG@10", "Recall@20"),
            ndcg_k=10, recall_k=20, **options):
    """Evaluate both rankings with the shared harness, then bootstrap each metric."""
    result_a = evaluate(predictions_a, labels, ndcg_k, recall_k)
    result_b = evaluate(predictions_b, labels, ndcg_k, recall_k)
    counts = {key: (result_a[key], result_b[key])
              for key in ("total_feeds", "evaluated_feeds", "excluded_zero_positive_feeds")}
    if any(a != b for a, b in counts.values()):
        raise ValueError(f"feed counts differ between rankings: {counts}")
    return {
        "total_feeds": result_a["total_feeds"],
        "evaluated_feeds": result_a["evaluated_feeds"],
        "excluded_zero_positive_feeds": result_a["excluded_zero_positive_feeds"],
        "comparisons": [paired_bootstrap(result_a, result_b, metric, **options) for metric in metrics],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True, help="Predictions of ranking A")
    parser.add_argument("--b", type=Path, required=True, help="Predictions of ranking B")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--unit", choices=("user", "feed"), default="user")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(_load(args.a, "predictions"), _load(args.b, "predictions"),
                         _load(args.labels, "labels"),
                         metrics=(f"NDCG@{args.ndcg_k}", f"Recall@{args.recall_k}"),
                         ndcg_k=args.ndcg_k, recall_k=args.recall_k, unit=args.unit,
                         resamples=args.resamples, seed=args.seed, alpha=args.alpha)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Bootstrap error: {error}\n")
    report = {"a": args.name_a, "b": args.name_b, "labels": str(args.labels), **report}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
