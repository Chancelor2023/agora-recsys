"""Rebuild every local artifact from the raw IDEA files, in order.

    python -m agora.build --idea-dir data/raw/Datasets/IDEA/dataset

1. data pipeline  -> data/processed/ (feeds, train reads, separate labels, articles)
2. encode         -> artifacts/models/embeddings.npz (+ saved TF-IDF/SVD encoder)
3. train          -> artifacts/models/model.json (logistic baseline) + loss curves
4. train-neural   -> artifacts/models/neural.pt (NRMS-style network, served by the API) + loss curves
Test labels are never read. Takes about two minutes on a laptop.
Optional, needs internet: python -m agora.data.live  (today's articles for the live mode)
"""
import argparse
import subprocess
import sys
import time


def step(title, *args):
    print(f"\n=== {title} ===", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, "-m", *args], check=True)
    print(f"--- {title}: {time.perf_counter() - start:.1f}s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idea-dir", required=True, help="Folder with IDEA's articles/interactions/recommendations.json")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--models", default="artifacts/models")
    parser.add_argument("--curves", default="reports/week3/loss_curves.png")
    parser.add_argument("--neural-curves", default="reports/week3/neural_loss_curves.png")
    args = parser.parse_args(argv)
    d, m = args.data, args.models
    step("1/4 data pipeline", "agora.data.pipeline", "--idea-dir", args.idea_dir, "--out", d)
    step("2/4 article embeddings", "agora.models", "encode", "--articles", f"{d}/articles.json",
         "--output", f"{m}/embeddings.npz")
    step("3/4 logistic baseline training", "agora.models", "train", "--config", "configs/experiment.toml",
         "--articles", f"{d}/articles.json", "--reads", f"{d}/reads_train.csv",
         "--embeddings", f"{m}/embeddings.npz", "--model", f"{m}/model.json",
         "--feeds", f"{d}/feeds_train.jsonl", "--labels", f"{d}/train_labels.json",
         "--validation-feeds", f"{d}/feeds_validation.jsonl",
         "--validation-labels", f"{d}/validation_labels.json",
         "--report", f"{m}/training_report.json", "--curves", args.curves)
    step("4/4 neural recommender training", "agora.models", "train-neural", "--config", "configs/experiment.toml",
         "--articles", f"{d}/articles.json", "--reads", f"{d}/reads_train.csv",
         "--embeddings", f"{m}/embeddings.npz", "--model", f"{m}/neural.pt",
         "--feeds", f"{d}/feeds_train.jsonl", "--labels", f"{d}/train_labels.json",
         "--validation-feeds", f"{d}/feeds_validation.jsonl",
         "--validation-labels", f"{d}/validation_labels.json",
         "--report", f"{m}/neural_report.json", "--curves", args.neural_curves)
    print("\nDone. Optional, today's articles:  python -m agora.data.live\n"
          "Start the API:  python -m uvicorn agora.api.main:app\n"
          "Then the interface:   python -m streamlit run src/agora/frontend/app.py")


if __name__ == "__main__":
    main()
