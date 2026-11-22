"""Tests du protocole temporel (docs/temporal_protocol.md) sur données synthétiques."""
import pandas as pd
import pytest

from agora.data.pipeline import (
    BOUNDS, add_windows, assign_split, build_labels, clean_feeds, to_utc,
)
from agora.data.idea import normalize_topic


def utc(*values):
    return pd.Series(pd.to_datetime(list(values), utc=True))


FAR = pd.Timestamp("2024-01-01", tz="UTC")  # fin des observations lointaine


# ---------------------------------------------------------------- fuseaux / splits
def test_bounds_are_local_midnight():
    assert BOUNDS["train"][1] == pd.Timestamp("2023-11-29 23:00", tz="UTC")
    assert BOUNDS["validation"][0] == BOUNDS["train"][1]   # pas de trou
    assert BOUNDS["test"][0] == BOUNDS["validation"][1]


def test_to_utc_naive_is_amsterdam():
    out = to_utc(pd.Series(["2023-11-30 00:30:00"]))
    assert out.iloc[0] == pd.Timestamp("2023-11-29 23:30", tz="UTC")


def test_to_utc_keeps_explicit_timezone():
    out = to_utc(pd.Series(["2023-11-29T23:30:00Z"]))
    assert out.iloc[0] == pd.Timestamp("2023-11-29 23:30", tz="UTC")


def test_assign_split_uses_local_midnight():
    s = assign_split(utc("2023-11-29 22:59:59",   # 23:59:59 locale -> train
                         "2023-11-29 23:00:00",   # 00:00 locale le 30 -> validation
                         "2023-11-29 23:30:00",   # piège UTC -> validation
                         "2023-12-07 23:00:00",   # fin du test, exclue [a, b)
                         "2023-11-22 22:59:59"))  # avant le train
    assert s.iloc[0] == "train"
    assert s.iloc[1] == "validation"
    assert s.iloc[2] == "validation"
    assert pd.isna(s.iloc[3]) and pd.isna(s.iloc[4])


# ---------------------------------------------------------------- nettoyage des feeds
def test_clean_feeds():
    raw = pd.DataFrame({
        "feed_id": ["f1", "f1", "f2", "f3"],
        "user_id": ["u1", "u1", "u1", "u1"],
        "generated_at": ["2023-11-24 10:00:00", "2023-11-24 11:00:00",
                         "2023-11-24 12:00:00", "2023-12-20 12:00:00"],
        "candidates": ['["a1", "a2", "a1", " a2 "]', "a9", "[]", "a1"],
    })
    st = {}
    f = clean_feeds(raw, st)
    assert list(f.feed_id) == ["f1"]                      # f2 vide, f3 hors période
    assert f.candidates.iloc[0] == ["a1", "a2"]           # doublons retirés, ordre gardé
    assert f.ts.iloc[0] == pd.Timestamp("2023-11-24 09:00", tz="UTC")  # 1re occurrence
    assert st["feeds_duplicate_feed_id"] == 1
    assert st["candidates_duplicate_removed"] == 2
    assert st["feeds_empty_candidates"] == 1
    assert st["feeds_outside_periods"] == 1


# ---------------------------------------------------------------- fenêtres
@pytest.fixture
def feeds():
    return pd.DataFrame({
        "feed_id": ["f1", "f2", "f3"],
        "user_id": ["u1", "u1", "u2"],
        "ts": utc("2023-11-24 10:00", "2023-11-24 14:00", "2023-11-29 22:00"),
        "split": ["train", "train", "train"],
        "candidates": [["a1", "a2", "a3"], ["a2", "a4", "a5"], ["a1", "a6"]],
    })


def _by_id(f):
    return f.set_index("feed_id")


def test_window_cut_by_next_feed(feeds):
    f = _by_id(add_windows(feeds, FAR, {}))
    assert f.loc["f1", "window_end"] == pd.Timestamp("2023-11-24 14:00", tz="UTC")
    assert f.loc["f1", "cut_by_next_feed"]
    assert f.loc["f2", "window_end"] == pd.Timestamp("2023-11-25 14:00", tz="UTC")  # +24 h
    assert not f.loc["f2", "cut_by_next_feed"]


def test_window_cut_by_period_end(feeds):
    f = _by_id(add_windows(feeds, FAR, {}))
    assert f.loc["f3", "window_end"] == BOUNDS["train"][1]   # 23:00 UTC, pas +24 h
    assert f.loc["f3", "cut_by_period_end"] and f.loc["f3", "shortened_window"]


def test_window_cut_by_data_end(feeds):
    obs_end = pd.Timestamp("2023-11-24 20:00", tz="UTC")
    f = _by_id(add_windows(feeds, obs_end, {}))
    assert f.loc["f2", "window_end"] == obs_end
    assert f.loc["f2", "cut_by_data_end"]


def test_same_timestamp_feeds_give_empty_window(feeds):
    feeds.loc[1, "ts"] = feeds.loc[0, "ts"]
    st = {}
    add_windows(feeds, FAR, st)
    assert st["feeds_empty_window"] >= 1


# ---------------------------------------------------------------- labels
def test_build_labels(feeds):
    f = add_windows(feeds, FAR, {})
    first_reads = pd.DataFrame({
        "user_id":  ["u1", "u1", "u1", "u1", "u2"],
        "article_id": ["a1", "a2", "a4", "a5", "a6"],
        "first_read_ts": pd.to_datetime([
            "2023-11-24 11:00",   # a1 : dans la fenêtre de f1
            "2023-11-24 15:00",   # a2 : après f2 -> attribué à f2, pas à f1
            "2023-11-25 15:00",   # a4 : f2 + 25 h -> hors 24 h
            "2023-11-24 09:00",   # a5 : lu AVANT le feed
            "2023-11-29 23:30",   # a6 : après la fin du train
        ], utc=True),
    })
    c = build_labels(f, first_reads)
    lab = {(r.feed_id, r.article_id): r.label for r in c.itertuples()}
    assert lab == {
        ("f1", "a1"): 1, ("f1", "a2"): 0, ("f1", "a3"): 0,
        ("f2", "a2"): 1, ("f2", "a4"): 0, ("f2", "a5"): 0,
        ("f3", "a1"): 0,   # lu par u1, pas par u2 : pas de fuite entre lecteurs
        ("f3", "a6"): 0,
    }
    assert len(c) == 8   # le merge ne duplique pas de lignes


def test_label_boundaries_half_open(feeds):
    f = add_windows(feeds, FAR, {})
    first_reads = pd.DataFrame({
        "user_id": ["u1", "u1"], "article_id": ["a1", "a2"],
        "first_read_ts": utc("2023-11-24 10:00",    # = ts du feed -> inclus
                             "2023-11-24 14:00"),   # = window_end de f1 -> exclu
    })
    c = build_labels(f[f.feed_id == "f1"], first_reads)
    lab = dict(zip(c.article_id, c.label))
    assert lab["a1"] == 1 and lab["a2"] == 0


# ---------------------------------------------------------------- topics
@pytest.mark.parametrize("bad", [None, "", "   ", 42, float("nan")])
def test_normalize_topic_fallback(bad):
    assert normalize_topic(bad) == "other"


def test_normalize_topic_unknown():
    assert normalize_topic("zzz-unknown-category") == "other"


def test_sky_news_home_is_not_lifestyle():
    assert normalize_topic("home") not in ("lifestyle", "other")


def test_normalize_topic_html_and_case():
    assert normalize_topic("Arts & Culture") == normalize_topic("arts & culture")