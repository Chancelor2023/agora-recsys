"""Adapter from the raw IDEA files to the pipeline's generic tables.

IDEA (https://github.com/Informfully/Datasets/tree/main/IDEA) ships one row per
recommended article. A *feed* is one recommendation list: all rows sharing
(informfully_id, createdAt). Reads come from interactions.json.

Timestamps: the text timestamps of IDEA were produced by the official R script
with as.POSIXct() and no explicit timezone, i.e. in the processing machine's
local time. Measured against views.json (raw epoch milliseconds, UTC), the
interaction timestamps are exactly 1 hour ahead of UTC for 34,338 of 34,871
matched reads: Central European Time (UTC+1, no daylight saving in Nov-Dec
2023). SOURCE_TZ = "Europe/Amsterdam" encodes that offset.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

SOURCE_TZ = "Europe/Amsterdam"

# Coarse topics used by the relevance model (topic affinity) and the reranker
# agents. IDEA's primaryCategory has 300+ outlet-specific values.
TOPIC_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("uk_news", ("royal",)),  # before lifestyle's "family"
    ("environment", ("environment", "climate", "cop28", "home insulation", "energy industry",
                     "coastal erosion", "plastic", "sewage", "hydrogen", "electricity",
                     "marine biology", "volcano", "volcanoes", "earth science", "europe heatwaves", "snow",
                     "met office")),
    ("politics", ("politics", "autumn statement", "brexit", "prime minister", "conservative party",
                  "labour party", "keir starmer", "james cleverly", "robert jenrick", "david cameron",
                  "nigel farage", "tony blair", "us election", "us congress", "us supreme court",
                  "leo varadkar", "home office", "immigration", "caroline dinenage", "stella creasy",
                  "stephen kinnock", "alex cunningham", "alex salmond", "ulster unionist",
                  "ministry of defence", "department for", "inquiry", "council", "kim jong un",
                  "aukus", "human rights", "competition and markets")),
    ("world", ("world", "us-news", "us news", "australia-news", "israel", "gaza", "palestin",
               "asia", "global-development", "global", "middle east", "russia", "ukraine",
               "europe", "americas", "africa", "india", "pakistan", "japan", "france", "germany",
               "italy", "spain", "greece", "turkey", "malta", "netherlands", "georgia", "ghana",
               "sierra leone", "south korea", "vietnam", "philippines", "malaysia", "xinjiang",
               "hong kong", "myanmar", "argentina", "delhi", "dublin", "paris", "barcelona",
               "abu dhabi", "new york", "michigan", "colorado", "united states", "usa",
               "australia", "central america", "fbi", "jihadism")),
    ("sport", ("sport", "football", "ufc", "rugby", "boxing", "snooker", "cricket", "darts",
               "cycling", "athletics", "nfl", "nba", "motor racing", "olympics", "mountaineering",
               "women's", "men's")),
    ("culture", ("entertainment", "tv", "radio", "television", "books", "stage", "theatre", "film",
                 "music", "games", "gaming", "media", "podcast", "exhibition", "photography", "art",
                 "culture", "comedy", "opera", "dance", "booker", "turner prize", "eurovision",
                 "strictly", "x factor", "i'm a celebrity", "doctor who", "brit awards", "bbc",
                 "itv", "reviews", "reality", "festival", "afrobeats", "billie eilish",
                 "nicki minaj", "rod stewart", "roald dahl", "claudia winkleman", "spotify",
                 "official uk music charts", "mercury prize", "live music", "christmas music")),
    ("business", ("business", "money", "companies", "savings", "personal finance", "personal debt",
                  "housing", "property", "mortgages", "renting", "inflation", "tax", "retail",
                  "hsbc", "amazon", "banking", "retailing", "pensioners", "unemployment", "flexible working",
                  "cryptocurrency", "nfts", "deals", "where to live")),
    ("tech_science", ("technology", "science", "space", "artificial intelligence", "gadgets",
                      "cyber", "data breaches", "exoplanets", "archaeology", "bacteria",
                      "social media", "tiktok", "elon musk", "car technology", "smart",
                      "innovate-with-azure")),
    ("health", ("health", "nhs", "mental health", "depression", "covid", "coronavirus", "measles",
                "asthma", "endometriosis", "hiv", "vaping", "fentanyl", "doctors", "social care",
                "cosmetic surgery")),
    ("crime", ("crime", "police", "constabulary", "metropolitan police", "violence", "abuse",
               "attacks", "dog attacks", "organised crime", "high court", "law", "spying", "mi5",
               "lorry deaths", "oscar pistorius", "disappearance", "george floyd", "graffiti",
               "gun", "sexual violence", "bullying")),
    ("opinion", ("opinion", "commentisfree", "comment", "voices", "theobserver", "diary")),
    ("lifestyle", ("lifeandstyle", "life", "food", "drink", "wine", "travel", "holidays", "wellness",
                   "beauty", "skincare", "makeup", "hair", "fashion", "shoes", "accessories",
                   "fitness", "black friday", "christmas", "shopping", "dating", "wedding", "home",
                   "furniture", "interiors", "bedroom", "bathroom", "kitchen", "appliances",
                   "garden", "gardening", "pets", "kids", "family", "inspiration", "inspiring", "restaurants",
                   "going out", "celebrity homes", "chef", "bath and body", "styled", "motoring",
                   "rail travel", "ibiza", "cotswolds", "winter sun", "red wine", "white wine",
                   "generation z", "features")),
    ("uk_news", ("uk news", "uk-news", "news", "royal", "london", "wales", "society", "education",
                 "teaching", "universities", "transport", "manchester", "birmingham", "berkshire",
                 "swindon", "hull", "huyton", "shrewsbury", "mansfield", "harlow", "watford",
                 "dudley", "eastbourne", "leigh", "oban", "haverfordwest", "immingham",
                 "rainworth", "waterbeach", "newtown", "baberton", "maida vale", "chelsea",
                 "reading", "caerphilly", "aberdeenshire", "thurrock", "potholes", "poverty",
                 "winter survival", "hillsborough", "falklands", "english civil war",
                 "world war two", "lgbt", "abortion", "100 women", "child", "civil rights")),
]


def normalize_topic(category) -> str:
    """Map a raw IDEA primaryCategory to one of the coarse topics, else 'other'."""
    if not isinstance(category, str) or not category.strip():
        return "other"
    text = html.unescape(category).strip().lower()
    for topic, keys in TOPIC_RULES:
        for key in keys:
            if text == key or re.search(r"(?<![a-z])" + re.escape(key) + r"s?(?![a-z])", text):
                return topic
    return "other"


def _read_json(path: Path) -> list[dict]:
    # articles.json mixes encodings: decode as UTF-8 and replace stray bytes.
    return json.loads(path.read_bytes().decode("utf-8", errors="replace"))


def load_idea(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return (feeds, reads, articles, provenance) from the raw IDEA directory."""
    raw = Path(raw_dir)
    recs = pd.DataFrame(_read_json(raw / "recommendations.json"))
    inter = pd.DataFrame(_read_json(raw / "interactions.json"))
    arts = pd.DataFrame(_read_json(raw / "articles.json"))
    info = {"recommendation_rows": len(recs), "interaction_rows": len(inter),
            "article_rows": len(arts)}

    # Feeds: one list per (reader, createdAt), candidates in the served order.
    recs = recs.sort_values(["informfully_id", "createdAt", "prediction"],
                            ascending=[True, True, False], kind="stable")
    stamp = recs.createdAt.str.replace(r"[^0-9]", "", regex=True)
    recs["feed_id"] = recs.informfully_id + "_" + stamp
    feeds = (recs.groupby("feed_id", sort=False)
             .agg(user_id=("informfully_id", "first"), generated_at=("createdAt", "first"),
                  candidates=("articleId", list), algorithm=("recommendationAlgorithm", "first"),
                  experimental_week=("experimental_week", "first"))
             .reset_index())
    info["feeds"] = len(feeds)

    reads = inter.rename(columns={"informfully_id": "user_id", "articleId": "article_id",
                                  "createdAt": "read_at"})[["user_id", "article_id", "read_at"]]

    info["article_duplicate_ids"] = int(arts.articleId.duplicated().sum())
    arts = arts.drop_duplicates("articleId", keep="first")
    published = pd.to_datetime(arts.datePublished, errors="coerce")
    published = published.dt.tz_localize(SOURCE_TZ, ambiguous="NaT", nonexistent="NaT")
    articles = pd.DataFrame({
        "article_id": arts.articleId.astype(str).str.strip(),
        "title": arts.title.fillna("").map(html.unescape).str.strip(),
        "lead": arts.lead.fillna("").map(html.unescape).str.strip(),
        "topic": arts.primaryCategory.map(normalize_topic),
        "category": arts.primaryCategory.fillna("").map(html.unescape),
        "outlet": arts.outlet.fillna(""),
        "published_at": published.dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": arts.url.fillna(""),
        "image": arts.image.fillna(""),
        "flag": arts.flag.fillna("none"),
    })
    info["articles"] = len(articles)
    info["topics"] = articles.topic.value_counts().to_dict()
    return feeds, reads, articles, info
