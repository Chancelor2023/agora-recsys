"""Pipeline de préparation des données AGORA (V1).

Applique le protocole temporel (docs/temporal_protocol.md) :
- splits [début, fin) : train 23-29/11, validation 30/11-02/12, test 03-07/12 (2023)
- label = 1 si la PREMIÈRE lecture (user, article) d'un candidat tombe dans
  [génération du feed, min(prochain feed du lecteur, +24h, fin de période, fin des observations))
- lecteurs retenus : >= 5 articles distincts lus pendant l'entraînement
- labels exportés séparément des entrées du modèle.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

# ============================ CONFIG (À ADAPTER) ============================
FEED_COLS = {"feed_id": "feed_id", "user_id": "user_id",
             "ts": "generated_at", "candidates": "candidates"}
READ_COLS = {"user_id": "user_id", "article_id": "article_id", "ts": "read_at"}
SOURCE_TZ = "UTC"   # fuseau appliqué aux timestamps bruts SANS fuseau
PERIOD_TZ = "UTC"   # fuseau dans lequel les dates du plan sont exprimées
SPLITS = {"train": ("2023-11-23", "2023-11-30"),
          "validation": ("2023-11-30", "2023-12-03"),
          "test": ("2023-12-03", "2023-12-08")}
WINDOW = pd.Timedelta(hours=24)
MIN_TRAIN_ARTICLES = 5
# ===========================================================================

BOUNDS = {k: (pd.Timestamp(a, tz=PERIOD_TZ).tz_convert("UTC"),
              pd.Timestamp(b, tz=PERIOD_TZ).tz_convert("UTC"))
          for k, (a, b) in SPLITS.items()}


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".jsonl", ".json"):
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            return pd.read_json(p)
    return pd.read_csv(p, sep="\t" if p.suffix == ".tsv" else ",")


def norm_id(s: pd.Series) -> pd.Series:
    """Normalisation unique des identifiants, partagée par tous les fichiers."""
    return s.astype("string").str.strip()


def to_utc(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(SOURCE_TZ, ambiguous="NaT", nonexistent="NaT")
    return ts.dt.tz_convert("UTC")


def parse_candidates(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("["):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                return ast.literal_eval(x)
        return [c for c in x.replace(";", ",").split(",") if c.strip()]
    return []


def assign_split(ts: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    for name, (a, b) in BOUNDS.items():
        out[(ts >= a) & (ts < b)] = name
    return out


# --------------------------------------------------------------------------
def clean_reads(raw: pd.DataFrame, st: dict) -> pd.DataFrame:
    r = raw.rename(columns={v: k for k, v in READ_COLS.items()})[list(READ_COLS)]
    st["reads_raw"] = len(r)
    r = r.dropna(subset=["user_id", "article_id", "ts"])
    st["reads_after_dropna"] = len(r)
    r["user_id"], r["article_id"] = norm_id(r.user_id), norm_id(r.article_id)
    r["ts"] = to_utc(r["ts"])
    st["reads_bad_timestamp"] = int(r.ts.isna().sum())
    r = r.dropna(subset=["ts"]).drop_duplicates()
    st["reads_after_dedup"] = len(r)
    st["reads_min_ts"], st["reads_max_ts"] = str(r.ts.min()), str(r.ts.max())
    return r.sort_values("ts")


def clean_feeds(raw: pd.DataFrame, st: dict) -> pd.DataFrame:
    f = raw.rename(columns={v: k for k, v in FEED_COLS.items()})[list(FEED_COLS)]
    st["feeds_raw"] = len(f)
    f = f.dropna(subset=["feed_id", "user_id", "ts"])
    f["feed_id"], f["user_id"] = norm_id(f.feed_id), norm_id(f.user_id)
    f["ts"] = to_utc(f["ts"])
    st["feeds_bad_timestamp"] = int(f.ts.isna().sum())
    f = f.dropna(subset=["ts"])
    st["feeds_duplicate_feed_id"] = int(f.feed_id.duplicated().sum())
    f = f.drop_duplicates("feed_id", keep="first")

    f["candidates"] = f["candidates"].map(parse_candidates)
    n_before = f["candidates"].map(len).sum()
    f["candidates"] = f["candidates"].map(
        lambda c: list(dict.fromkeys(str(a).strip() for a in c if str(a).strip())))
    st["candidates_duplicate_removed"] = int(n_before - f["candidates"].map(len).sum())
    st["feeds_empty_candidates"] = int((f["candidates"].map(len) == 0).sum())
    f = f[f["candidates"].map(len) > 0]

    f["split"] = assign_split(f.ts)
    st["feeds_outside_periods"] = int(f.split.isna().sum())
    return f.dropna(subset=["split"])


def add_windows(f: pd.DataFrame, obs_end: pd.Timestamp, st: dict) -> pd.DataFrame:
    f = f.sort_values(["user_id", "ts"]).copy()
    f["next_ts"] = f.groupby("user_id")["ts"].shift(-1)
    cap = f.ts + WINDOW
    f["period_end"] = f.split.map(lambda s: BOUNDS[s][1])
    ends = pd.DataFrame({"cap": cap, "next": f.next_ts.fillna(cap),
                         "period": f.period_end, "obs": obs_end})
    f["window_end"] = ends.min(axis=1)
    f["cut_by_next_feed"] = f.next_ts.notna() & (f.next_ts < cap)
    f["cut_by_period_end"] = (f.period_end < cap) & (f.window_end == f.period_end)
    f["cut_by_data_end"] = (obs_end < cap) & (f.window_end == obs_end)
    f["shortened_window"] = f.cut_by_period_end | f.cut_by_data_end
    f["empty_window"] = f.window_end <= f.ts
    for c in ["cut_by_next_feed", "cut_by_period_end", "cut_by_data_end",
              "shortened_window", "empty_window"]:
        st[f"feeds_{c}"] = int(f[c].sum())
    return f


def build_labels(f: pd.DataFrame, first_reads: pd.DataFrame) -> pd.DataFrame:
    c = (f[["feed_id", "user_id", "ts", "window_end", "split", "candidates"]]
         .explode("candidates").rename(columns={"candidates": "article_id"}))
    c["article_id"] = norm_id(c.article_id)
    c = c.merge(first_reads, on=["user_id", "article_id"], how="left")
    c["label"] = ((c.first_read_ts >= c.ts) & (c.first_read_ts < c.window_end)).astype(int)
    return c


# --------------------------------------------------------------------------
def run(feeds_path, reads_path, out_dir, sample_feeds=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    st = {"source_tz": SOURCE_TZ, "period_tz": PERIOD_TZ,
          "splits": {k: [str(a), str(b)] for k, (a, b) in BOUNDS.items()}}

    reads = clean_reads(read_any(reads_path), st)
    feeds = clean_feeds(read_any(feeds_path), st)

    # Lecteurs éligibles : >= 5 articles distincts lus en train
    tr_a, tr_b = BOUNDS["train"]
    reads_train = reads[(reads.ts >= tr_a) & (reads.ts < tr_b)]
    n_art = reads_train.groupby("user_id").article_id.nunique()
    eligible = set(n_art[n_art >= MIN_TRAIN_ARTICLES].index)
    st["users_with_train_reads"] = int(len(n_art))
    st["users_eligible"] = len(eligible)
    st["feeds_before_user_filter"] = len(feeds)
    feeds = feeds[feeds.user_id.isin(eligible)]
    st["feeds_after_user_filter"] = len(feeds)

    if sample_feeds:
        feeds = feeds.groupby("split", group_keys=False).head(sample_feeds)

    # Première lecture (sur toutes les données, pour ne pas la confondre avec une relecture)
    first_reads = (reads.groupby(["user_id", "article_id"], as_index=False).ts.min()
                   .rename(columns={"ts": "first_read_ts"}))
    obs_end = reads.ts.max() + pd.Timedelta(microseconds=1)
    feeds = add_windows(feeds, obs_end, st)
    cand = build_labels(feeds, first_reads)

    # Contrôles de correspondance des identifiants
    known = set(reads.article_id)
    st["candidates_total"] = len(cand)
    st["candidates_never_read_by_anyone"] = int((~cand.article_id.isin(known)).sum())
    cand_ids = set(cand.article_id)
    st["reads_articles_never_candidate"] = int((~reads.article_id.isin(cand_ids)).sum())

    # -------- Exports --------
    # Entrées du modèle (SANS labels, SANS lectures val/test)
    reads_train.assign(ts=reads_train.ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")) \
        .to_csv(out / "reads_train.csv", index=False)
    json.dump(sorted(eligible), open(out / "eligible_users.json", "w"))
    for split in SPLITS:
        fs = feeds[feeds.split == split]
        fs.assign(generated_at=fs.ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))[
            ["feed_id", "user_id", "generated_at", "candidates"]
        ].to_json(out / f"feeds_{split}.jsonl", orient="records", lines=True)

        cs = cand[cand.split == split]
        # train_labels -> Ronald (entraînement du scorer) ; validation/test -> Jerry uniquement
        cs[["feed_id", "article_id", "label"]].to_json(
            out / f"{split}_labels.json", orient="records", lines=True)

        g = cs.groupby("feed_id").label.sum()
        st[split] = {"feeds": int(fs.feed_id.nunique()), "users": int(fs.user_id.nunique()),
                     "candidates": len(cs), "positives": int(cs.label.sum()),
                     "feeds_without_positive": int((g == 0).sum()),
                     "feeds_shortened_window": int(fs.shortened_window.sum())}

    feeds[["feed_id", "user_id", "split", "ts", "window_end", "cut_by_next_feed",
           "cut_by_period_end", "cut_by_data_end", "empty_window"]] \
        .to_csv(out / "feed_windows_report.csv", index=False)

    first = feeds[feeds.split == "train"].head(1)
    first.assign(generated_at=first.ts.astype(str))[
        ["feed_id", "user_id", "generated_at", "candidates"]
    ].to_json(out / "first_feed_sample.json", orient="records", indent=2)

    json.dump(st, open(out / "cleaning_report.json", "w"), indent=2, default=str)
    print(json.dumps(st, indent=2, default=str))
    return st


def main():
    p = argparse.ArgumentParser(description="AGORA data pipeline V1")
    p.add_argument("--feeds", required=True)
    p.add_argument("--reads", required=True)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--sample-feeds", type=int, default=None,
                   help="N feeds par split (test rapide)")
    a = p.parse_args()
    run(a.feeds, a.reads, a.out, a.sample_feeds)


if __name__ == "__main__":
    main()