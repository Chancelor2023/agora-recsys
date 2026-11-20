"""Ingest today's articles from the outlets of IDEA, through their public RSS feeds.

    python -m agora.data.live --sources configs/live_sources.json --out data/live/articles_live.json

The model scores an article from its text, topic and outlet, so recent articles can
be recommended without retraining. Each item becomes an article with the same
fields as data/processed/articles.json (article_id, title, lead, topic, category,
outlet, published_at UTC, url, image) plus source and fetched_at. A feed that fails
is skipped with a warning; the run fails only if no article at all is collected.
Use --from-file to parse a saved RSS file (tests, offline demo).
"""
from __future__ import annotations

import argparse
import email.utils
import gzip
import hashlib
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .idea import normalize_topic

USER_AGENT = "Agora-recsys/0.2 (student project; RSS reader)"
NS = {"media": "http://search.yahoo.com/mrss/", "dc": "http://purl.org/dc/elements/1.1/",
      "atom": "http://www.w3.org/2005/Atom", "content": "http://purl.org/rss/1.0/modules/content/"}
TAG_RE = re.compile(r"<[^>]+>")


def _text(node, path):
    found = node.find(path, NS)
    return (found.text or "").strip() if found is not None and found.text else ""


def _clean(value: str, limit: int = 400) -> str:
    value = html.unescape(TAG_RE.sub(" ", html.unescape(value or "")))
    value = re.sub(r"\s+", " ", value).strip()
    return value[:limit]


def _parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)  # RFC 822 (RSS)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # ISO 8601 (Atom, dc:date)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _image(item) -> str:
    for path in ("media:thumbnail", "media:content", "enclosure"):
        node = item.find(path, NS)
        if node is not None and node.get("url"):
            return node.get("url")
    return ""


def parse_feed(xml_bytes: bytes, source: dict, fetched_at: datetime) -> list[dict]:
    """Parse one RSS 2.0 or Atom document into article records."""
    root = ET.fromstring(xml_bytes)
    items = root.findall(".//item") or root.findall(".//atom:entry", NS)
    articles = []
    for item in items:
        title = _clean(_text(item, "title") or _text(item, "atom:title"), 300)
        link = _text(item, "link") or _text(item, "guid")
        if not link:
            atom_link = item.find("atom:link", NS)
            link = atom_link.get("href", "") if atom_link is not None else ""
        if not title or not link:
            continue
        lead = _clean(_text(item, "description") or _text(item, "atom:summary")
                      or _text(item, "content:encoded"))
        published = _parse_date(_text(item, "pubDate") or _text(item, "dc:date")
                                or _text(item, "atom:published") or _text(item, "atom:updated"))
        categories = [c.text.strip() for c in item.findall("category") if c.text and c.text.strip()]
        topic = source.get("topic")
        if not topic:
            topics = [normalize_topic(c) for c in categories]
            topic = next((t for t in topics if t != "other"), "uk_news")
        articles.append({
            "article_id": "live-" + hashlib.sha1(link.encode("utf-8")).hexdigest()[:16],
            "title": title, "lead": lead or title, "topic": topic,
            "category": categories[0] if categories else source.get("topic", ""),
            "outlet": source["outlet"],
            "published_at": (published or fetched_at).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "url": link, "image": _image(item), "source": source["url"],
            "fetched_at": fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    return articles


def fetch(url: str, timeout: float = 10.0) -> bytes:
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "gzip, identity",
        "Accept": "application/rss+xml, application/xml, text/xml, */*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (public RSS)
        data = response.read()
    if data[:2] == b"\x1f\x8b":  # gzip-compressed body (some outlets send it regardless)
        data = gzip.decompress(data)
    return data.lstrip()


def collect(sources: list[dict], max_age_hours: float = 48.0, fetcher=fetch,
            now: datetime | None = None, log=print) -> list[dict]:
    """Fetch every source; keep articles younger than max_age_hours, one per URL."""
    now = now or datetime.now(timezone.utc)
    by_url: dict[str, dict] = {}
    for source in sources:
        try:
            items = parse_feed(fetcher(source["url"]), source, now)
        except Exception as error:  # one broken feed must not stop the others
            log(f"warning: skipped {source['url']} ({error.__class__.__name__}: {error})")
            continue
        kept = 0
        for article in items:
            age = now - datetime.fromisoformat(article["published_at"].replace("Z", "+00:00"))
            if age > timedelta(hours=max_age_hours) or article["url"] in by_url:
                continue
            by_url[article["url"]] = article
            kept += 1
        log(f"{source['outlet']:<16} {source.get('topic') or 'auto':<13} {kept:3d} articles  {source['url']}")
    return sorted(by_url.values(), key=lambda a: a["published_at"], reverse=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default="configs/live_sources.json")
    parser.add_argument("--out", default="data/live/articles_live.json")
    parser.add_argument("--max-age-hours", type=float, default=48.0)
    parser.add_argument("--from-file", help="Parse a saved RSS file instead of fetching")
    parser.add_argument("--outlet", default="BBC", help="Outlet name for --from-file")
    args = parser.parse_args(argv)
    if args.from_file:
        sources = [{"url": args.from_file, "outlet": args.outlet}]
        fetcher = lambda path: Path(path).read_bytes()  # noqa: E731
        articles = collect(sources, max_age_hours=1e9, fetcher=fetcher)
    else:
        config = json.loads(Path(args.sources).read_text(encoding="utf-8"))
        articles = collect(config["sources"], args.max_age_hours)
    if not articles:
        sys.exit("No article collected: check the network connection and configs/live_sources.json")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(articles, indent=1, ensure_ascii=False), encoding="utf-8")
    topics = {}
    for a in articles:
        topics[a["topic"]] = topics.get(a["topic"], 0) + 1
    print(f"Saved {len(articles)} articles to {out}: {dict(sorted(topics.items(), key=lambda t: -t[1]))}")


if __name__ == "__main__":
    main()
