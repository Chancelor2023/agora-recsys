# IDEA data
Brett owns acquisition, cleaning and storage.

Official source: https://github.com/Informfully/Datasets/tree/main/IDEA (public; files used: `articles.json`, `interactions.json`, `recommendations.json`).

## Get the data
```
git clone --depth 1 https://github.com/Informfully/Datasets.git data/raw/Datasets
python -m agora.data.pipeline --idea-dir data/raw/Datasets/IDEA/dataset --out data/processed
```
`data/raw/` and `data/processed/` are ignored by Git. Do not upload participant-level records.

## Timestamps
IDEA's text timestamps were written by the official R script with `as.POSIXct()` and no explicit timezone. Compared with `views.json` (raw epoch milliseconds, UTC), interaction timestamps are one hour ahead of UTC (rounded to the hour) for 34,763 of 34,871 matched reads (99.7%): Central European Time, UTC+1 throughout Nov–Dec 2023, with no clock change. Reproduce with `python -m agora.data.verify_timezone --idea-dir data/raw/Datasets/IDEA/dataset`. `source_timezone = "Europe/Amsterdam"` in `configs/experiment.toml`; split boundaries are local midnights in that zone. The first read in the data is 23 Nov 2023 00:00:09 local time, the experiment start.

## Mapping to the pipeline
- Feed = all recommendation rows sharing (`informfully_id`, `createdAt`), candidates in served order (`prediction` descending); `feed_id` = `<user>_<timestamp>`.
- Read = one row of `interactions.json`.
- Article topic = IDEA's 300+ `primaryCategory` values mapped to 12 coarse topics (`src/agora/data/idea.py`; 10 articles left as `other`).

## Outputs (`data/processed/`)
| File | Content | Used by |
| --- | --- | --- |
| `feeds_{train,validation,test}.jsonl` | feed_id, user_id, generated_at (UTC), candidates | model, reranker, API |
| `reads_train.csv` | training-period reads only | profiles, popularity |
| `articles.json` | article_id, title, lead, topic, category, outlet, published_at, url | model, reranker, API |
| `train_labels.json` | first-read labels, train split | model training |
| `validation_labels.json`, `test_labels.json` | separate labels | validation: model selection; test: final evaluation only |
| `cleaning_report.json`, `feed_windows_report.csv` | counts and attribution windows | audit |

## Full run on IDEA (run on 19 Nov 2026)
207,220 recommendation rows → 7,972 feeds (1,809 duplicate candidates removed); 34,890 reads; 10,948 articles (6 duplicate IDs removed). 558 of 590 readers have ≥ 5 distinct training reads; 7,593 feeds kept.

| Split | Feeds | Readers | Candidates | Positives | Feeds without positive |
| --- | --- | --- | --- | --- | --- |
| Train | 3,317 | 558 | 86,216 | 15,325 | 797 |
| Validation | 1,620 | 558 | 41,622 | 6,202 | 514 |
| Test | 2,656 | 531 | 67,786 | 8,293 | 1,144 |

No candidate or read is missing from the article catalogue. 1,620 attribution windows are shortened by a split or data boundary (flagged in `feed_windows_report.csv`).

## Today's articles (live mode)
`python -m agora.data.live` reads the public RSS feeds of IDEA's outlets listed in `configs/live_sources.json` and writes `data/live/articles_live.json` with the same fields as `articles.json`. Each feed's topic is set in the config, or inferred from the items' categories with the same 12-topic mapping. Articles older than 48 hours are dropped, a failing feed is skipped. These articles are only served by the API (`/recommend_live`); they are never used for training or evaluation.
