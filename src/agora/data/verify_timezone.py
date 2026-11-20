"""Check which timezone IDEA's text timestamps are in.

    python -m agora.data.verify_timezone --idea-dir data/raw/Datasets/IDEA/dataset

interactions.json stores reads as local text ("2023-11-23 13:57:32"); views.json
stores page views as raw epoch milliseconds (UTC). For each read, the closest
"Article" page view of the same reader and article gives the offset between the
two clocks (the view follows the read by a few seconds). On IDEA (commit 9a71a3e),
rounded to the hour, the offset is +1 h for almost every matched read: Central
European Time (UTC+1), as set in configs/experiment.toml.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def offsets(idea_dir):
    root = Path(idea_dir)
    reads = pd.DataFrame(json.loads((root / "interactions.json").read_text(encoding="utf-8")))
    views = pd.DataFrame(json.loads((root / "views.json").read_text(encoding="utf-8")))
    views = views[(views.page == "Article") & views.articleId.notna()].copy()
    views["utc"] = pd.to_datetime(views.createdAt.astype("int64"), unit="ms")
    reads["local"] = pd.to_datetime(reads.createdAt)
    reads["read_key"] = range(len(reads))
    pairs = reads.merge(views[["informfully_id", "articleId", "utc"]], on=["informfully_id", "articleId"])
    pairs["hours"] = (pairs.local - pairs.utc).dt.total_seconds() / 3600
    closest = pairs.loc[pairs.groupby("read_key").hours.apply(lambda s: s.abs().idxmin())]
    return closest.hours, len(reads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idea-dir", required=True)
    args = parser.parse_args(argv)
    hours, total = offsets(args.idea_dir)
    whole = hours.round().astype(int)
    one = int((whole == 1).sum())
    print(f"reads: {total}, matched with a page view: {len(hours)}")
    print(f"offset rounded to the hour = +1 h: {one} ({one / len(hours):.2%})")
    print(f"within 6 minutes of +1 h: {int(((hours - 1).abs() <= 0.1).sum())}")
    print("offsets rounded to the hour:", whole.value_counts().to_dict())


if __name__ == "__main__":
    main()
