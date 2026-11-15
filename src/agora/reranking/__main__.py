"""Run with python -m agora.reranking --input ... --config ... --output ...

Reranks every feed of the input file in chronological order with the
SCRUF-D-inspired reranker, updating the exposure state between feeds, and
writes predictions in the evaluation format (user_id, feed_id, article_id,
rank, relevance_score) plus the selected agent and final state.
Use --lambda 0 for the ablation (relevance ranking only).
"""

import argparse
import json
from pathlib import Path

from .io import build_reranker, load_feeds, load_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/reranking_agents.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lambda", dest="lam", type=float)
    parser.add_argument("--k", type=int)
    parser.add_argument("--allocation", choices=("lottery", "weighted"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    try:
        config = load_json(args.config)
        reranker = build_reranker(config, lam=args.lam, k=args.k,
                                  allocation=args.allocation, seed=args.seed)
        source = load_json(args.input)
        feeds = load_feeds(args.input)
        results = reranker.rerank_feeds(feeds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Reranking error: {error}\n")

    output = {
        "is_synthetic": bool(source.get("is_synthetic", False)) if isinstance(source, dict) else False,
        "method": "scruf_d" if reranker.lam > 0 else "relevance_only_ablation",
        "settings": {"lambda": reranker.lam, "k": reranker.k,
                     "allocation": reranker.allocation, "window": reranker.state.window,
                     "agents": config["agents"]},
        "predictions": [row for r in results for row in r.predictions()],
        "selections": [r.summary() for r in results],
        "final_state": reranker.state.to_dict(),
    }

    for r in results:
        top = ", ".join(f"{row['article_id']}({row.get('topic', '?')})"
                        for row in r.ranking[: reranker.k])
        print(f"{r.feed_id}: selected_agent={r.selected_agent} "
              f"fairness_before={r.agent_fairness} exposure={r.exposure}")
        print(f"  top-{reranker.k}: {top}")
    print(f"OK: {len(results)} feeds reranked, {len(output['predictions'])} ranked rows, "
          f"state updated {reranker.state.n_updates} times.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"Saved to {args.output}")
    return output


if __name__ == "__main__":
    main()
