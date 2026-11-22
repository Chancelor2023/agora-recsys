"""Week 3 / final experiment: every ranking on one split, with the same evaluator.

    python -m agora.experiment --split validation
    python -m agora.experiment --split test        # only after python -m agora.freeze create

Rankings: random, topic popularity (training reads), logistic baseline, neural
(NRMS-style), neural + SCRUF-D. Metrics: NDCG@10 and Recall@20 (Jerry's harness),
IPS-NDCG@10 (position-debiased), and top-10 exposure of each protected topic
versus its share of the candidates. Also a lambda sweep, the accuracy/fairness
trade-off plot and three reader examples before/after reranking.
Writes reports/week3/experiment_<split>.{json,md} and tradeoff_<split>.png.

On the test split: the frozen manifest must verify (python -m agora.freeze verify),
only lambda in {0, selected} is computed, --lambdas is refused, and a first result is
never overwritten (a re-run needs --rerun-reason, writes a new file and is logged).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from agora.evaluation import evaluate
from agora.evaluation.ips import estimate_propensity, ips_ndcg
from agora.models.io import file_hash, load_feeds, read_records
from agora.models.relevance import FeedPredictor, RelevanceModel, rank_candidates
from agora.reranking.io import build_reranker
from agora.reranking.state import position_weight

# Closed list of rows reported on the test split (frozen in the manifest).
# Order and names follow docs/test_protocol.md, section 3.
TEST_ROWS = ["Random", "Most popular (article)", "Topic popularity", "Logistic regression",
             "Neural (NRMS-style)", "Neural + SCRUF-D"]
VALIDATION_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 1.0]
# Declared comparisons (docs/test_protocol.md, section 5): A minus B, paired bootstrap over readers.
COMPARISONS = [("Neural (NRMS-style)", "Random"), ("Neural (NRMS-style)", "Most popular (article)"),
               ("Neural (NRMS-style)", "Topic popularity"), ("Neural (NRMS-style)", "Logistic regression"),
               ("Neural + SCRUF-D", "Neural (NRMS-style)")]


def _rows(path):
    return read_records(path, "labels")


def _uniform(seed, feed_id, article_id):
    """Deterministic number in [0, 1) for one (feed, article)."""
    return random.Random(int(hashlib.sha256(f"{seed}:{feed_id}:{article_id}".encode()).hexdigest(), 16)).random()


def _ranking(feed, scores):
    return rank_candidates(feed["user_id"], feed["feed_id"], feed["candidates"], scores)


def _exposure(predictions, topic_of, k=10):
    exposure, total = Counter(), 0.0
    for row in predictions:
        if row["rank"] <= k:
            w = position_weight(row["rank"])
            exposure[topic_of[row["article_id"]]] += w
            total += w
    return {t: v / total for t, v in exposure.items()}


def _preflight_test(args):
    """Every check of a test run, before anything is logged: --lambdas refused, frozen state
    intact (the manifest is the only freeze mechanism), no earlier run without --rerun-reason."""
    from agora.freeze import manifest_path, run_log_path, sha256, verify

    if args.lambdas:
        raise SystemExit("--lambdas is refused on the test split: only 0 and the frozen lambda run")
    problems = verify(args.data, args.models, args.config, args.rerank_config)
    if problems:
        raise SystemExit("Frozen state check failed, test not run:\n- " + "\n- ".join(problems))
    out = Path(args.out)
    log = run_log_path(args.models)
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    previous = [r for r in map(json.loads, lines) if r["status"] == "started"]
    stem = "experiment_test"
    if previous or (out / f"{stem}.json").exists():
        if not args.rerun_reason:
            raise SystemExit("A test run already exists. A re-run needs --rerun-reason (evaluator bug or "
                             "failed run with no result); it is logged and both runs are published.")
        stem = f"experiment_test_run{len(previous) + 1}"
    return {"run": len(previous) + 1, "manifest_sha256": sha256(manifest_path(args.models)), "log": str(log),
            "reason": args.rerun_reason, "stem": stem}


def _log_test_start(plan):
    """Log the run as started: called once every input is loaded, just before ranking."""
    from datetime import datetime, timezone

    record = {**plan, "started_at": datetime.now(timezone.utc).isoformat(), "status": "started"}
    log = Path(record["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def _start_test_run(args):
    return _log_test_start(_preflight_test(args))


def _finish_test_run(record, result_path):
    from datetime import datetime, timezone
    from agora.freeze import sha256

    done = {**record, "status": "finished", "finished_at": datetime.now(timezone.utc).isoformat(),
            "result": str(result_path), "result_sha256": sha256(result_path)}
    with Path(record["log"]).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(done) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--models", default="artifacts/models")
    parser.add_argument("--config", default="configs/experiment.toml")
    parser.add_argument("--rerank-config", default="configs/reranking_agents.json")
    parser.add_argument("--out", default="reports/week3")
    parser.add_argument("--lambdas", default=None,
                        help="Validation only: comma-separated sweep (default 0,0.1,0.25,0.5,1)")
    parser.add_argument("--rerun-reason", default=None,
                        help="Test only: why a second test run is allowed (evaluator bug, failed run)")
    args = parser.parse_args(argv)

    from agora.models.__main__ import load_context
    from agora.models.neural import NeuralRecommender, _group_popularity, build_examples

    d, m, split = Path(args.data), Path(args.models), args.split
    ctx = argparse.Namespace(config=args.config, articles=str(d / "articles.json"),
                             reads=str(d / "reads_train.csv"), embeddings=str(m / "embeddings.npz"))
    test_plan = _preflight_test(args) if split == "test" else None  # before loading anything
    config, builder, _, _ = load_context(ctx)
    run_record = None
    feeds = load_feeds(d / f"feeds_{split}.jsonl")
    feeds.sort(key=lambda f: (pd.Timestamp(f["generated_at"]), f["user_id"], f["feed_id"]))
    labels = _rows(d / f"{split}_labels.json")
    propensity = estimate_propensity(load_feeds(d / "feeds_train.jsonl"), _rows(d / "train_labels.json"))
    articles = builder.articles
    topic_of = articles.topic.to_dict()
    title_of = articles.title.to_dict() if "title" in articles else {}

    # --- base rankings ------------------------------------------------------
    rankings = {}
    rankings["Random"] = [r for f in feeds for r in _ranking(
        f, [_uniform(42, f["feed_id"], a) for a in f["candidates"]])]
    # Article-level popularity: no candidate was read in training, so every score is 0 and the
    # order is the article-ID tie-break ("oldest first"). Reported as such, see the protocol.
    train_reads = builder.reads[builder.reads.ts < builder.protocol.train_end].article_id.value_counts()
    rankings["Most popular (article)"] = [r for f in feeds for r in _ranking(
        f, [float(train_reads.get(a, 0)) for a in f["candidates"]])]
    # Topic popularity with a seeded random tie-break between articles of the same topic
    # (1e-9 is far below the gap between two topic shares).
    topic_pop, _ = _group_popularity(builder, builder.protocol.train_end)
    rankings["Topic popularity"] = [r for f in feeds for r in _ranking(
        f, [topic_pop.get(topic_of[a], 0.0) + 1e-9 * _uniform("tie-42", f["feed_id"], a)
            for a in f["candidates"]])]
    logistic, _ = RelevanceModel.load(m / "model.json")
    predictor = FeedPredictor(builder, logistic)
    rankings["Logistic regression"] = [r for f in feeds for r in predictor.predict_feed(
        f["user_id"], f["feed_id"], f["generated_at"], f["candidates"], split)]
    neural, payload = NeuralRecommender.load(m / "neural.pt", builder)
    examples, _ = build_examples(builder, feeds, split, max_history=neural.cfg.max_history)
    scored = {e["feed_id"]: s for e, s in zip(examples, neural.score(examples))}
    rankings["Neural (NRMS-style)"] = [r for f in feeds for r in _ranking(f, scored[f["feed_id"]])]
    rerank_config = json.loads(Path(args.rerank_config).read_text(encoding="utf-8"))
    build_reranker(rerank_config, lam=0.0)  # a broken reranker config fails here, before the log
    if test_plan:
        run_record = _log_test_start(test_plan)  # all inputs loaded and scored: the run starts

    # --- SCRUF-D on top of the neural model, lambda sweep ---------------------
    by_feed = {}
    for row in rankings["Neural (NRMS-style)"]:
        by_feed.setdefault(row["feed_id"], []).append(row)
    sweep = {}
    chosen = float(rerank_config.get("lambda", 0.25))
    if split == "test":
        lambdas = sorted({0.0, chosen})  # ablation and the frozen choice, nothing else
    else:
        lambdas = [float(x) for x in args.lambdas.split(",")] if args.lambdas else VALIDATION_LAMBDAS
    for lam in lambdas:
        reranker = build_reranker(rerank_config, lam=lam)  # its own exposure state
        reranked = []
        for f in feeds:
            cands = [{"article_id": r["article_id"], "relevance_score": r["relevance_score"],
                      "topic": topic_of[r["article_id"]]} for r in by_feed[f["feed_id"]]]
            result = reranker.rerank({"user_id": f["user_id"], "feed_id": f["feed_id"], "candidates": cands})
            reranked += [{**p, "relevance_score": p["relevance_score"]} for p in result.predictions()]
        sweep[lam] = reranked
    rankings["Neural + SCRUF-D"] = sweep[chosen]
    assert list(rankings) == TEST_ROWS

    # --- metrics -----------------------------------------------------------
    agents = rerank_config["agents"]
    # Supplementary public-affairs metrics at 5 (docs/temporal_protocol.md): only reads of the
    # first agent's protected topics count as relevant; feeds without such a read are excluded.
    focus = agents[0]
    focus_topics = set(focus["protected_values"])
    focus_labels = [{**r, "label": int(r["label"] and topic_of[r["article_id"]] in focus_topics)} for r in labels]
    pool = Counter(topic_of[a] for f in feeds for a in f["candidates"])
    pool_share = {t: c / sum(pool.values()) for t, c in pool.items()}

    evaluated = {}

    def metrics(preds, record=None):
        clean = [{k: v for k, v in r.items() if k in ("user_id", "feed_id", "article_id", "rank", "relevance_score")}
                 for r in preds]
        result = evaluate(clean, labels, 10, 20)
        if record is not None:
            evaluated[record] = result
        exposure = _exposure(preds, topic_of)
        out = {"NDCG@10": result["metrics"]["NDCG@10"], "Recall@20": result["metrics"]["Recall@20"],
               "IPS-NDCG@10": ips_ndcg(preds, labels, feeds, propensity),
               "evaluated_feeds": result["evaluated_feeds"], "total_feeds": result["total_feeds"],
               "excluded_zero_positive_feeds": result["excluded_zero_positive_feeds"]}
        for agent in agents:
            topics = set(agent["protected_values"])
            out[f"exposure_{agent['name']}"] = sum(v for t, v in exposure.items() if t in topics)
        name = focus["name"]
        at5 = evaluate(clean, focus_labels, 5, 5)
        out[f"NDCG@5_{name}"] = at5["metrics"]["NDCG@5"]
        out[f"Recall@5_{name}"] = at5["metrics"]["Recall@5"]
        out[f"evaluated_feeds_{name}"] = at5["evaluated_feeds"]
        out[f"exposure@5_{name}"] = sum(v for t, v in _exposure(preds, topic_of, k=5).items() if t in focus_topics)
        return out

    # Personalisation check: readers who received the same candidates should not all
    # get the same top 10. Mean Jaccard overlap of top-10 sets between such readers.
    import itertools
    same_pool = {}
    for f in feeds:
        same_pool.setdefault(frozenset(f["candidates"]), []).append(f["feed_id"])
    groups = [g[:30] for g in same_pool.values() if len(g) >= 5]

    def overlap(preds):
        top = {}
        for row in preds:
            if row["rank"] <= 10:
                top.setdefault(row["feed_id"], set()).add(row["article_id"])
        pairs = [len(top[a] & top[b]) / len(top[a] | top[b])
                 for g in groups for a, b in itertools.combinations(g, 2)]
        return sum(pairs) / len(pairs) if pairs else None

    table = {name: metrics(preds, record=name) for name, preds in rankings.items()}
    from agora.evaluation.bootstrap import paired_bootstrap
    comparisons = [{"a": a, "b": b, **{metric: {k: v for k, v in paired_bootstrap(
                        evaluated[a], evaluated[b], metric, unit="user", resamples=2000, seed=42).items()
                        if k in ("difference_a_minus_b", "ci_low", "ci_high", "units")}
                    for metric in ("NDCG@10", "Recall@20")}} for a, b in COMPARISONS]
    for name, preds in rankings.items():
        table[name]["top10_overlap_same_candidates"] = overlap(preds)
    sweep_table = {lam: metrics(preds) for lam, preds in sweep.items()}
    targets = {a["name"]: {"target_share": a["target_share"],
                           "candidate_share": sum(pool_share.get(t, 0) for t in a["protected_values"])}
               for a in agents}

    # --- three reader examples ---------------------------------------------
    before = {f["feed_id"]: by_feed[f["feed_id"]] for f in feeds}
    after = {}
    for row in sweep[chosen]:
        after.setdefault(row["feed_id"], []).append(row)
    protected = set(agents[0]["protected_values"])

    def moved(fid):
        old = {r["article_id"]: r["rank"] for r in before[fid]}
        return max([old[r["article_id"]] - r["rank"] for r in after[fid]
                    if topic_of[r["article_id"]] in protected and r["rank"] <= 10] or [0])
    picked, users = [], set()
    for fid in sorted(after, key=moved, reverse=True):
        uid = after[fid][0]["user_id"]
        day = after[fid][0]["feed_id"].rsplit("_", 1)[-1][:8]
        if uid not in users and day not in users and moved(fid) > 0:
            picked.append(fid)
            users.update({uid, day})  # three different readers on three different days
        if len(picked) == 3:
            break
    reads_of = builder.reads.groupby("user_id").article_id.apply(list)
    examples_md = []
    for fid in picked:
        uid = after[fid][0]["user_id"]
        hist = Counter(topic_of[a] for a in reads_of.get(uid, []))
        old = {r["article_id"]: r["rank"] for r in before[fid]}
        lines = [f"### Reader `{uid}`, feed `{fid}`",
                 f"Training history: {sum(hist.values())} reads, mostly "
                 + ", ".join(f"{t} ({c})" for t, c in hist.most_common(3)) + ".", "",
                 "| Rank with SCRUF-D | Rank without | Article | Topic |", "| --- | --- | --- | --- |"]
        for r in sorted(after[fid], key=lambda r: r["rank"])[:10]:
            a = r["article_id"]
            lines.append(f"| {r['rank']} | {old[a]} | {title_of.get(a, a)[:70]} | {topic_of[a]} |")
        lines.append("")
        examples_md.append("\n".join(lines))

    # --- outputs -----------------------------------------------------------
    out = Path(args.out)
    stem = run_record["stem"] if run_record else f"experiment_{split}"
    out.mkdir(parents=True, exist_ok=True)
    report = {"split": split, "feeds": len(feeds), "neural_model": payload["model_version"],
              "lambda_selected": chosen, "lambdas": lambdas,
              "manifest_sha256": run_record["manifest_sha256"] if run_record else None,
              "propensity_by_position": [round(p, 3) for p in propensity],
              "targets": targets, "rankings": table, "comparisons": comparisons,
              "lambda_sweep": sweep_table, "examples": picked}
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    names = [a["name"] for a in agents]
    first = next(iter(table.values()))
    md = [f"# Experiment on the {split} split ({len(feeds)} feeds)", "",
          f"Evaluated feeds: {first['evaluated_feeds']} of {first['total_feeds']} "
          f"({first['excluded_zero_positive_feeds']} excluded: no read in the attribution window). "
          f"Neural + SCRUF-D uses lambda {chosen:g}.", "",
          "Generated by `python -m agora.experiment --split " + split + "`. "
          "IPS-NDCG weights each read by 1 / P(seen | displayed position) (propensities from the training split). "
          "Top-10 overlap: mean Jaccard overlap of the top 10 between readers who received the same candidates "
          f"({len(groups)} shared candidate sets); 1 = everyone gets the same list.", "",
          "| Ranking | NDCG@10 | Recall@20 | IPS-NDCG@10 | Top-10 overlap between readers | " + " | ".join(f"Top-10 exposure {n}" for n in names) + " |",
          "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for name, row in table.items():
        md.append(f"| {name} | {fmt(row['NDCG@10'])} | {fmt(row['Recall@20'])} | {fmt(row['IPS-NDCG@10'])} | "
                  f"{fmt(row['top10_overlap_same_candidates'])} | "
                  + " | ".join(f"{row['exposure_' + n]:.3f}" for n in names) + " |")
    fn = focus["name"]
    md += ["", f"Supplementary {fn} metrics at 5 (only reads of {', '.join(sorted(focus_topics))} count as relevant; "
           f"{first[f'evaluated_feeds_{fn}']} feeds with at least one such read):", "",
           f"| Ranking | NDCG@5 {fn} | Recall@5 {fn} | Top-5 exposure {fn} |", "| --- | --- | --- | --- |"]
    for name, row in table.items():
        md.append(f"| {name} | {fmt(row[f'NDCG@5_{fn}'])} | {fmt(row[f'Recall@5_{fn}'])} | {row[f'exposure@5_{fn}']:.3f} |")
    md += ["", "Declared comparisons (A − B; paired bootstrap over readers, 2,000 resamples, seed 42, 95% CI):", "",
           "| A | B | NDCG@10 | Recall@20 |", "| --- | --- | --- | --- |"]
    for c in comparisons:
        md.append(f"| {c['a']} | {c['b']} | " + " | ".join(
            f"{c[m]['difference_a_minus_b']:+.4f} [{c[m]['ci_low']:+.4f}, {c[m]['ci_high']:+.4f}]"
            for m in ("NDCG@10", "Recall@20")) + " |")
    md += ["", "Targets: " + "; ".join(f"{n} {t['target_share']:.3f} (share of {split} candidates {t['candidate_share']:.3f})"
                                       for n, t in targets.items()) + ".", "",
           "## Lambda sweep (neural + SCRUF-D)", "",
           "| lambda | NDCG@10 | IPS-NDCG@10 | " + " | ".join(f"Exposure {n}" for n in names) + " |",
           "| --- | --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for lam, row in sweep_table.items():
        md.append(f"| {lam:g} | {fmt(row['NDCG@10'])} | {fmt(row['IPS-NDCG@10'])} | "
                  + " | ".join(f"{row['exposure_' + n]:.3f}" for n in names) + " |")
    md += ["", f"![Trade-off](tradeoff_{split}.png)", "", "## Three readers before / after SCRUF-D", ""] + examples_md
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lams = list(sweep_table)
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
        xs = [sweep_table[l]["NDCG@10"] for l in lams]
        ys = [sweep_table[l][f"exposure_{names[0]}"] for l in lams]
        ax.plot(xs, ys, marker="o", color="#1f77b4")
        seen = {}
        for l, x, y in zip(lams, xs, ys):
            key = (round(x, 4), round(y, 4))
            seen[key] = seen.get(key, []) + [f"{l:g}"]
        for (x, y), ls in seen.items():
            ax.annotate("λ=" + ", ".join(ls), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.axhline(targets[names[0]]["target_share"], color="#d62728", linestyle="--", linewidth=1,
                   label=f"target ({names[0]})")
        ax.set_xlabel("NDCG@10")
        ax.set_ylabel(f"top-10 exposure of {names[0]}")
        ax.set_title(f"Accuracy / fairness trade-off ({split})")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"tradeoff_{split}.png")
        plt.close(fig)
    except ImportError:
        pass
    print("\n".join(md[:7 + len(table)]))
    if run_record:
        _finish_test_run(run_record, out / f"{stem}.json")
    print(f"\nSaved {out}/{stem}.md, .json and tradeoff_{split}.png")


if __name__ == "__main__":
    main()
