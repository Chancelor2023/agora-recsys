"""End-to-end checks of the freeze / single-test protocol (docs/test_protocol.md).

A tiny synthetic dataset is built once (articles, train reads, feeds and labels of
the three splits, TF-IDF embeddings, logistic and neural models), frozen, then
copied for every test. ``python -m agora.experiment --split test`` runs on the
synthetic "test" split only: no IDEA data is read.

The safeguards first written as ``xfail(strict=True)`` at commit b2f6342 are now
enforced by the freeze manifest (``python -m agora.freeze``) and pass.

If the freeze command or the experiment command changes, edit only ``freeze`` and
``run_experiment`` below.
"""
import argparse
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ["politics", "sport", "environment", "culture"]
FAVOURITE = {"u1": "politics", "u2": "sport", "u3": "environment", "u4": "culture"}


# --------------------------------------------------------------------------- #
# Commands under test: the only two functions to adapt when the CLI changes.
# --------------------------------------------------------------------------- #
def freeze(root):
    """Freeze everything that determines the test results."""
    for model in ("model.json", "neural.pt"):
        subprocess.run([sys.executable, "-m", "agora.models", "freeze",
                        "--model", str(root / "models" / model),
                        "--validation-note", "synthetic freeze-protocol test"],
                       check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "agora.freeze", "create",
                    "--note", "synthetic freeze-protocol test", "--idea-dir", str(root / "no-idea"),
                    "--data", str(root / "data"), "--models", str(root / "models"),
                    "--config", str(root / "configs" / "experiment.toml"),
                    "--rerank-config", str(root / "configs" / "reranking_agents.json")],
                   check=True, capture_output=True)


def run_experiment(root, *extra):
    from agora.experiment import main
    with redirect_stdout(StringIO()):
        main(["--split", "test", "--data", str(root / "data"), "--models", str(root / "models"),
              "--config", str(root / "configs" / "experiment.toml"),
              "--rerank-config", str(root / "configs" / "reranking_agents.json"),
              "--out", str(root / "out"), *extra])
    return json.loads((root / "out" / "experiment_test.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #
def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _feeds_and_labels(split, days, candidates, topic_of, empty=()):
    """One feed per reader and day; a candidate is read when it has the reader's favourite topic."""
    feeds, labels = [], []
    for d, day in enumerate(days):
        for user, favourite in FAVOURITE.items():
            fid = f"{user}_{split}_{d}"
            feeds.append({"feed_id": fid, "user_id": user, "generated_at": day, "candidates": candidates})
            labels += [{"feed_id": fid, "article_id": a,
                        "label": int(fid not in empty and topic_of[a] == favourite)} for a in candidates]
    return feeds, labels


def _build(root):
    from agora.models.__main__ import _load_labels, _matrices, load_context
    from agora.models.embeddings import encode_articles
    from agora.models.io import load_articles
    from agora.models.neural import NeuralConfig, NeuralRecommender, build_examples
    from agora.models.relevance import RelevanceModel

    data, models, configs = root / "data", root / "models", root / "configs"
    for folder in (data, models, configs):
        folder.mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "experiment.toml", configs / "experiment.toml")
    shutil.copy(ROOT / "configs" / "reranking_agents.json", configs / "reranking_agents.json")

    words = ["budget", "match", "climate", "film", "vote", "goal", "flood", "album"]
    articles = []
    for i in range(40):
        topic = TOPICS[i % 4]
        articles.append({"article_id": f"a{i:02d}", "title": f"{topic} news story {words[i % 8]}",
                         "lead": f"{topic} report {words[(i + 3) % 8]}", "topic": topic,
                         "outlet": ["BBC", "GuardianInt"][i % 2], "published_at": "2023-11-22T08:00:00Z"})
    (data / "articles.json").write_text(json.dumps(articles), encoding="utf-8")
    topic_of = {a["article_id"]: a["topic"] for a in articles}

    reads = []  # 8 distinct training reads per reader (eligibility needs 5), before every feed
    for user, favourite in FAVOURITE.items():
        older = [a for a in topic_of if int(a[1:]) < 24]
        history = [a for a in older if topic_of[a] == favourite] + [a for a in older if topic_of[a] != favourite][:2]
        reads += [{"user_id": user, "article_id": a, "ts": f"2023-11-23T{9 + j:02d}:00:00.000000Z"}
                  for j, a in enumerate(history)]
    pd.DataFrame(reads).to_csv(data / "reads_train.csv", index=False)

    pools = {"train": [f"a{i:02d}" for i in range(24, 32)],
             "validation": [f"a{i:02d}" for i in range(28, 36)],
             "test": [f"a{i:02d}" for i in range(32, 40)]}
    days = {"train": [f"2023-11-{d}T12:00:00.000000Z" for d in (25, 26, 27, 28)],
            "validation": ["2023-12-01T12:00:00.000000Z"],
            "test": ["2023-12-04T12:00:00.000000Z", "2023-12-05T12:00:00.000000Z"]}
    for split in pools:
        empty = {"u4_test_1"} if split == "test" else set()  # one test feed without any read
        feeds, labels = _feeds_and_labels(split, days[split], pools[split], topic_of, empty)
        _write_jsonl(data / f"feeds_{split}.jsonl", feeds)
        _write_jsonl(data / f"{split}_labels.json", labels)

    encode_articles(load_articles(data / "articles.json"), models / "embeddings.npz", "tfidf")
    ctx = argparse.Namespace(config=str(configs / "experiment.toml"), articles=str(data / "articles.json"),
                             reads=str(data / "reads_train.csv"), embeddings=str(models / "embeddings.npz"))
    _, builder, binding, _ = load_context(ctx)
    from agora.models.io import load_feeds
    feeds = load_feeds(data / "feeds_train.jsonl")
    labels = _load_labels(data / "train_labels.json")
    matrices, targets, _ = _matrices(builder, feeds, labels, "train", True)
    logistic, report = RelevanceModel.fit(np.concatenate(matrices), targets, 1.0, 42, epochs=20)
    logistic.save(models / "model.json", {"binding": binding, "training_report": report})
    examples, _ = build_examples(builder, feeds, "train", labels, strict=True)
    neural = NeuralRecommender(builder, NeuralConfig(hidden=16, heads=2, epochs=2, batch_feeds=8))
    neural.fit(examples, None, log=lambda _: None)
    neural.save(models / "neural.pt", {"binding": binding})


@pytest.fixture(scope="module")
def frozen_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("frozen") / "run"
    _build(root)
    freeze(root)
    return root


@pytest.fixture
def run_dir(frozen_template, tmp_path):
    target = tmp_path / "run"
    shutil.copytree(frozen_template, target)
    return target


def _refused(root, *extra):
    with pytest.raises((SystemExit, ValueError)):
        run_experiment(root, *extra)


def _edit_json(path, change):
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Already enforced at b2f6342
# --------------------------------------------------------------------------- #
def test_untouched_frozen_artifacts_run(run_dir):
    report = run_experiment(run_dir)
    assert report["split"] == "test"
    assert report["feeds"] == 8


def test_refuses_unfrozen_models(run_dir):
    (run_dir / "models" / "neural.pt.frozen.json").unlink()
    _refused(run_dir)


def test_refuses_modified_model_file(run_dir):
    with (run_dir / "models" / "model.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")  # any byte change
    _refused(run_dir)


# --------------------------------------------------------------------------- #
# Inputs changed after the freeze: must be refused
# --------------------------------------------------------------------------- #
def _perturb_embeddings(root):
    path = root / "models" / "embeddings.npz"
    with np.load(path) as stored:
        ids, vectors = stored["ids"], stored["vectors"]
    with path.open("wb") as handle:  # same IDs and catalogue: passes load_embeddings
        np.savez_compressed(handle, ids=ids, vectors=vectors + 0.05)


def _append_bytes(name):
    def change(root):
        with (root / "models" / name).open("ab") as handle:
            handle.write(b"\0")
    return change


def _drop_one_read(root):
    path = root / "data" / "reads_train.csv"
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)


def _flip_one_train_label(root):
    path = root / "data" / "train_labels.json"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["label"] = 1 - rows[0]["label"]
    _write_jsonl(path, rows)


def _change_seed(root):
    path = root / "configs" / "experiment.toml"
    text = path.read_text(encoding="utf-8")
    assert "seed = 42" in text
    path.write_text(text.replace("seed = 42", "seed = 43"), encoding="utf-8")


def _change_lambda(root):
    _edit_json(root / "configs" / "reranking_agents.json", lambda c: c.__setitem__("lambda", 0.5))


def _change_target(root):
    _edit_json(root / "configs" / "reranking_agents.json",
               lambda c: c["agents"][0].__setitem__("target_share", 0.2))


TAMPERING = {
    "embedding vectors": _perturb_embeddings,
    "text encoder": _append_bytes("embeddings.npz.encoder.joblib"),
    "training reads": _drop_one_read,
    "training labels": _flip_one_train_label,
    "experiment config": _change_seed,
    "reranker lambda": _change_lambda,
    "reranker target": _change_target,
}


@pytest.mark.parametrize("change", list(TAMPERING), ids=list(TAMPERING))
def test_refuses_inputs_changed_after_freeze(run_dir, change):
    TAMPERING[change](run_dir)
    _refused(run_dir)


# --------------------------------------------------------------------------- #
# Single declared test run
# --------------------------------------------------------------------------- #
def test_only_frozen_lambda_and_ablation_on_test(run_dir):
    chosen = json.loads((run_dir / "configs" / "reranking_agents.json").read_text())["lambda"]
    report = run_experiment(run_dir)
    assert {float(lam) for lam in report.get("lambda_sweep", {})} <= {0.0, float(chosen)}


def test_lambda_sweep_option_refused_on_test(run_dir):
    _refused(run_dir, "--lambdas", "0,0.1,0.25,0.5,1")


def test_second_test_run_does_not_overwrite_results(run_dir):
    run_experiment(run_dir)
    _refused(run_dir)


def test_exclusion_counts_reported_for_every_ranking(run_dir):
    report = run_experiment(run_dir)
    counts = {(row["total_feeds"], row["evaluated_feeds"], row["excluded_zero_positive_feeds"])
              for row in report["rankings"].values()}
    assert counts == {(8, 7, 1)}
