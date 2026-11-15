"""Run with python -m agora.evaluation --predictions ... --labels ..."""
import argparse
import json
from pathlib import Path
from .harness import evaluate
from .config import load_cutoffs


def _load(path, key):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get(key) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list or an object containing {key}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--ndcg-k", type=int)
    parser.add_argument("--recall-k", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scope", choices=("primary", "supplementary"), default="primary")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        default_ndcg, default_recall = (load_cutoffs(args.config, args.scope)
                                        if args.config else ((5, 5) if args.scope == "supplementary" else (10, 20)))
        ndcg_k = args.ndcg_k if args.ndcg_k is not None else default_ndcg
        recall_k = args.recall_k if args.recall_k is not None else default_recall
        result = evaluate(_load(args.predictions, "predictions"),
                          _load(args.labels, "labels"),
                          ndcg_k, recall_k)
        result["evaluation_settings"] = {
            "config": str(args.config) if args.config else None,
            "scope": args.scope, "ndcg_k": ndcg_k, "recall_k": recall_k,
        }
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Evaluation error: {error}\n")
    print(text, end="")


if __name__ == "__main__":
    main()
