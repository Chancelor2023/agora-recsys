"""Week 3 / final experiment: every ranking on one split, with the same evaluator.

    python -m agora.experiment --split validation
    python -m agora.experiment --split test        # only after both models are frozen

Rankings: random, topic popularity (training reads), logistic baseline, neural
(NRMS-style), neural + SCRUF-D. Metrics: NDCG@10 and Recall@20 (Jerry's harness),
IPS-NDCG@10 (position-debiased), and top-10 exposure of each protected topic
versus its share of the candidates. Also a lambda sweep, the accuracy/fairness
trade-off plot and three reader examples before/after reranking.
Writes reports/week3/experiment_<split>.{json,md} and tradeoff_<split>.png.
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


def _rows(path):
    return read_records(path, "labels")


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


def _check_frozen(model_path):
    freeze = Path(str(model_path) + ".frozen.json")
    if not freeze.exists() or json.loads(freeze.read_text())["model_sha256"] != file_hash(model_path):
        raise SystemExit(f"{model_path} is not frozen: run python -m agora.models freeze first")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--models", default="artifacts/models")
    parser.add_argument("--config", default="configs/experiment.toml")
    parser.add_argument("--rerank-config", default="configs/reranking_agents.json")
    parser.add_argument("--out", default="reports/week3")
    parser.add_argument("--lambdas", default="0,0.1,0.25,0.5,1")
    args = parser.parse_args(argv)

    from agora.models.__main__ import load_context
    from agora.models.neural import NeuralRecommender, _group_popularity, build_examples

    d, m, split = Path(args.data), Path(args.models), args.split
    ctx = argparse.Namespace(config=args.config, articles=str(d / "articles.json"),
                             reads=str(d / "reads_train.csv"), embeddings=str(m / "embeddings.npz"))
    config, builder, _, _ = load_context(ctx)
    if split == "test":
        _check_frozen(m / "model.json")
        _check_frozen(m / "neural.pt")
    feeds = load_feeds(d / f"feeds_{split}.jsonl")
    feeds.sort(key=lambda f: (pd.Timestamp(f["generated_at"]), f["user_id"], f["feed_id"]))
    labels = _rows(d / f"{split}_labels.json")
    propensity = estimate_propensity(load_feeds(d / "feeds_train.jsonl"), _rows(d / "train_labels.json"))
    articles = builder.articles
    topic_of = articles.topic.to_dict()
    title_of = articles.title.to_dict() if "title" in articles else {}

    # --- base rankings ------------------------------------------------------
    rankings = {}
    rankings["Random"] = [r for f in feeds for r in _ranking(f, [
        random.Random(int(hashlib.sha256(f"42:{f['feed_id']}:{a}".encode()).hexdigest(), 16)).random()
        for a in f["candidates"]])]
    topic_pop, _ = _group_popularity(builder, builder.protocol.train_end)
    rankings["Topic popularity"] = [r for f in feeds for r in _ranking(
        f, [topic_pop.get(topic_of[a], 0.0) for a in f["candidates"]])]
    logistic, _ = RelevanceModel.load(m / "model.json")
    predictor = FeedPredictor(builder, logistic)
    rankings["Logistic regression"] = [r for f in feeds for r in predictor.predict_feed(
        f["user_id"], f["feed_id"], f["generated_at"], f["candidates"], split)]
    neural, payload = NeuralRecommender.load(m / "neural.pt", builder)
    examples, _ = build_examples(builder, feeds, split, max_history=neural.cfg.max_history)
    scored = {e["feed_id"]: s for e, s in zip(examples, neural.score(examples))}
    rankings["Neural (NRMS-style)"] = [r for f in feeds for r in _ranking(f, scored[f["feed_id"]])]

    # --- SCRUF-D on top of the neural model, lambda sweep ---------------------
    rerank_config = json.loads(Path(args.rerank_config).read_text(encoding="utf-8"))
    by_feed = {}
    for row in rankings["Neural (NRMS-style)"]:
        by_feed.setdefault(row["feed_id"], []).append(row)
    sweep = {}
    for lam in [float(x) for x in args.lambdas.split(",")]:
        reranker = build_reranker(rerank_config, lam=lam)  # its own exposure state
        reranked = []
        for f in feeds:
            cands = [{"article_id": r["article_id"], "relevance_score": r["relevance_score"],
                      "topic": topic_of[r["article_id"]]} for r in by_feed[f["feed_id"]]]
            result = reranker.rerank({"user_id": f["user_id"], "feed_id": f["feed_id"], "candidates": cands})
            reranked += [{**p, "relevance_score": p["relevance_score"]} for p in result.predictions()]
        sweep[lam] = reranked
    chosen = float(rerank_config.get("lambda", 0.25))
    rankings[f"Neural + SCRUF-D (lambda {chosen:g})"] = sweep[chosen]

    # --- metrics -----------------------------------------------------------
    agents = rerank_config["agents"]
    pool = Counter(topic_of[a] for f in feeds for a in f["candidates"])
    pool_share = {t: c / sum(pool.values()) for t, c in pool.items()}

    def metrics(preds):
        clean = [{k: v for k, v in r.items() if k in ("user_id", "feed_id", "article_id", "rank", "relevance_score")}
                 for r in preds]
        result = evaluate(clean, labels, 10, 20)
        exposure = _exposure(preds, topic_of)
        out = {"NDCG@10": result["metrics"]["NDCG@10"], "Recall@20": result["metrics"]["Recall@20"],
               "IPS-NDCG@10": ips_ndcg(preds, labels, feeds, propensity),
               "evaluated_feeds": result["evaluated_feeds"], "total_feeds": result["total_feeds"]}
        for agent in agents:
            topics = set(agent["protected_values"])
            out[f"exposure_{agent['name']}"] = sum(v for t, v in exposure.items() if t in topics)
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

    table = {name: metrics(preds) for name, preds in rankings.items()}
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
    out.mkdir(parents=True, exist_ok=True)
    report = {"split": split, "feeds": len(feeds), "neural_model": payload["model_version"],
              "propensity_by_position": [round(p, 3) for p in propensity],
              "targets": targets, "rankings": table, "lambda_sweep": sweep_table, "examples": picked}
    (out / f"experiment_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fmt = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    names = [a["name"] for a in agents]
    md = [f"# Experiment on the {split} split ({len(feeds)} feeds)", "",
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
    md += ["", "Targets: " + "; ".join(f"{n} {t['target_share']:.3f} (share of {split} candidates {t['candidate_share']:.3f})"
                                       for n, t in targets.items()) + ".", "",
           "## Lambda sweep (neural + SCRUF-D)", "",
           "| lambda | NDCG@10 | IPS-NDCG@10 | " + " | ".join(f"Exposure {n}" for n in names) + " |",
           "| --- | --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for lam, row in sweep_table.items():
        md.append(f"| {lam:g} | {fmt(row['NDCG@10'])} | {fmt(row['IPS-NDCG@10'])} | "
                  + " | ".join(f"{row['exposure_' + n]:.3f}" for n in names) + " |")
    md += ["", f"![Trade-off](tradeoff_{split}.png)", "", "## Three readers before / after SCRUF-D", ""] + examples_md
    (out / f"experiment_{split}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

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
    print(f"\nSaved {out}/experiment_{split}.md, .json and tradeoff_{split}.png")


if __name__ == "__main__":
    main()
