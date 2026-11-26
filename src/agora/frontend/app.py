"""Agora reader interface (Streamlit).

Run the API first, then:  python -m streamlit run src/agora/frontend/app.py --server.headless true
The API address can be changed with AGORA_API_URL (default http://127.0.0.1:8000).
Demo readers (names and habits) come from configs/demo_personas.json.
"""
import base64
import html
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("AGORA_API_URL", "http://127.0.0.1:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).resolve().parent / "assets"
LOGO, MARK = ASSETS / "agora_logo.png", ASSETS / "agora_mark.png"

# One colour per topic: the colour system of the whole interface.
TOPICS = {
    "politics": ("Politics", "#2F5BEA"), "environment": ("Environment", "#2E9E6B"),
    "world": ("World", "#7A5AF0"), "uk_news": ("UK news", "#4B9BD6"),
    "business": ("Business", "#7A6A58"), "sport": ("Sport", "#E8A92E"),
    "culture": ("Culture", "#9B4DCA"), "lifestyle": ("Lifestyle", "#F0A8C4"),
    "tech_science": ("Tech & science", "#159C9C"), "health": ("Health", "#D9708F"),
    "crime": ("Crime", "#8C2F39"), "opinion": ("Opinion", "#8A8F99"),
}
OTHER = ("Other", "#B5BCC8")

# Window on the world: how much Agora may step in (SCRUF-D lambda). "Balanced" is the
# setting evaluated on the test split; the other two are offered to the reader.
WINDOWS = {"Only my interests": 0.0, "Balanced": None, "Wide": 1.0}
WINDOW_HELP = ("Only my interests: ranked by your reading habits alone. "
               "Balanced: Agora keeps politics and environment news in view (the evaluated setting). "
               "Wide: Agora steps in more often.")

st.set_page_config(page_title="Agora", page_icon=str(MARK) if MARK.exists() else None, layout="wide")
if LOGO.exists():
    st.logo(str(LOGO), icon_image=str(MARK))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, .stApp, .stMarkdown, button, input, label, textarea, [data-testid="stSidebar"] {
  font-family: 'Bricolage Grotesque', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.ag-head { margin-bottom: 1.4rem; }
.ag-head img { height: 62px; width: auto; display: block; }
.ag-head h1 { font-size: 3.4rem; font-weight: 800; letter-spacing: -0.035em; line-height: 1; margin: 0; color: #1C1F26; }
.ag-head p { font-size: 1.15rem; color: #4A4F5A; margin: .8rem 0 0; max-width: 46rem; }
.ag-reader { background: #fff; border-radius: 12px; padding: 1rem 1.25rem 1.1rem; margin-bottom: 1.2rem; }
.ag-reader h3 { font-size: 1.35rem; font-weight: 700; margin: 0; color: #1C1F26; }
.ag-reader .sub { color: #6B7689; font-size: .92rem; margin: .15rem 0 0; }
.ag-strip { display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin: .7rem 0 .45rem; background: #ECECEC; }
.ag-strip span { display: block; height: 100%; }
.ag-strip span + span { box-shadow: inset 2px 0 0 #fff; }
.ag-legend { display: flex; flex-wrap: wrap; gap: .25rem 1.1rem; font-size: .86rem; color: #3A4457; }
.ag-legend i { display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .35rem; vertical-align: -1px; }
.ag-caption { font-size: .9rem; color: #6B7689; margin: .2rem 0 .8rem; }
.ag-item { display: grid; grid-template-columns: 6px 128px 1fr; column-gap: 1rem; background: #fff;
  border-radius: 10px; overflow: hidden; min-height: 112px; }
.ag-item.noimg { grid-template-columns: 6px 1fr; }
.ag-bar { background: var(--c); }
.ag-thumb { background: #ECECEC center / cover no-repeat; border-radius: 6px; margin: .8rem 0; }
.ag-body { padding: .75rem 1rem .85rem 0; }
.ag-top { font-size: .86rem; color: #6B7689; }
.ag-top b { color: var(--c); font-weight: 600; }
.ag-num { font-weight: 800; color: #1C1F26; margin-right: .5rem; }
.ag-title { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.2rem; line-height: 1.3;
  margin: .2rem 0 .3rem; color: #1C1F26; }
.ag-lead { font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem; line-height: 1.55; color: #3A4457;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; max-width: 62ch; }
.ag-why { display: inline-block; font-size: .84rem; margin-top: .5rem; padding: .12rem .6rem; border-radius: 999px;
  background: #EFEFEF; color: #4A4F5A; }
.ag-why.boost { background: #B2482E; color: #fff; }
.ag-row { display: flex; gap: .6rem; align-items: baseline; padding: .45rem .7rem; border-radius: 8px; background: #fff;
  margin-bottom: .3rem; border-left: 5px solid var(--c); }
.ag-row .t { font-family: 'Source Serif 4', Georgia, serif; font-size: .98rem; line-height: 1.35; color: #1C1F26; }
.ag-row .n { font-weight: 800; color: #9AA3B2; min-width: 1.4rem; }
.ag-row.new { background: #F6E6E1; }
.ag-row .tag { margin-left: auto; font-size: .78rem; color: #B2482E; font-weight: 600; white-space: nowrap; }
.ag-count { font-size: 2.4rem; font-weight: 800; color: #1C1F26; line-height: 1; }
.ag-count small { font-size: 1rem; font-weight: 400; color: #6B7689; margin-left: .4rem; }
[data-testid="stSidebar"] h2 { font-size: 1.05rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


def api(method: str, path: str, **kwargs):
    response = requests.request(method, API_URL + path, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def log_event(event_type: str, item: dict, data: dict) -> dict:
    return api("POST", "/log_interaction", json={
        "event_id": str(uuid.uuid4()), "user_id": data["user_id"], "feed_id": data["feed_id"],
        "article_id": item["article_id"], "event_type": event_type,
        "displayed_rank": item["rank"], "model_version": data["model_version"]})


def read_json(relative: str, default):
    try:
        return json.loads((ROOT / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def topic_of(key):
    return TOPICS.get(key or "", OTHER)


def esc(text) -> str:
    return html.escape(str(text or ""))


def strip(counts: Counter, legend: int = 6) -> str:
    """The topic mix as one coloured band, largest topic first, with a short legend."""
    total = sum(counts.values()) or 1
    ordered = counts.most_common()
    bars = "".join(f'<span style="width:{100 * n / total:.2f}%;background:{topic_of(t)[1]}" '
                   f'title="{esc(topic_of(t)[0])}: {n}"></span>' for t, n in ordered)
    keys = "".join(f'<span><i style="background:{topic_of(t)[1]}"></i>{esc(topic_of(t)[0])} '
                   f'{100 * n / total:.0f}%</span>' for t, n in ordered[:legend])
    return f'<div class="ag-strip">{bars}</div><div class="ag-legend">{keys}</div>'


def when(value) -> str:
    text = str(value or "")[:16].replace("T", " ")
    return text


# ---------------------------------------------------------------------------
brand = (f'<img src="data:image/png;base64,{base64.b64encode(LOGO.read_bytes()).decode()}" alt="Agora">'
         if LOGO.exists() else "<h1>Agora</h1>")
st.markdown(f'<div class="ag-head">{brand}<p>The news you like, with the stories that '
            'concern everyone kept in view.</p></div>', unsafe_allow_html=True)

try:
    mode = api("GET", "/mode")
except requests.RequestException as error:
    st.error(f"Agora cannot reach its service at {API_URL}. Start it with "
             f"`python -m uvicorn agora.api.main:app`, then reload this page "
             f"({error.__class__.__name__}).")
    st.stop()

if mode["is_mock"]:
    st.warning("Demo data only: download the models (`python scripts/fetch_models.py`) and restart the API.")
    st.stop()

agents = read_json("configs/reranking_agents.json", {}).get("agents", [])
protected = {v for a in agents for v in a.get("protected_values", [])}
personas = [p for p in read_json("configs/demo_personas.json", {}).get("personas", [])]
users = api("GET", "/users")
personas = [p for p in personas if p["user_id"] in users]

with st.sidebar:
    st.header("Who is reading?")
    labels = [f"{p['name']}, {p['habit']}" for p in personas] + ["Another reader"]
    choice = st.radio("Reader", labels, label_visibility="collapsed")
    if choice == "Another reader" or not personas:
        user_id = st.selectbox("Reader ID", users)
        name, habit = "This reader", ""
    else:
        persona = personas[labels.index(choice)]
        user_id, name, habit = persona["user_id"], persona["name"], persona["habit"]

    st.header("News")
    live = api("GET", "/live/status")
    sources = (["Today's news", "Archive, Nov–Dec 2023"] if live.get("available")
               else ["Archive, Nov–Dec 2023"])
    source = st.radio("News", sources, label_visibility="collapsed")
    if source == "Today's news":
        feed_id = "live"
        st.caption(f"{live['articles']} articles from 6 UK outlets, fetched {when(live.get('fetched_at'))} UTC")
    else:
        feeds = api("GET", f"/users/{user_id}/feeds")
        feed_labels = {f["feed_id"]: when(f["generated_at"]) for f in feeds}
        feed_id = st.selectbox("Morning edition", list(feed_labels), format_func=feed_labels.get,
                               key=f"feed-{user_id}")
        if not live.get("available"):
            st.caption("Today's news is off: run `python -m agora.data.live` to fetch it.")

    st.header("Window on the world")
    window = st.select_slider("Window", list(WINDOWS), value="Balanced",
                              label_visibility="collapsed", help=WINDOW_HELP)
    top_k = st.slider("Articles shown", 5, 20, 10)
    refresh = st.button("Refresh the selection", width="stretch")

lam = WINDOWS[window]

# Keep the list that was served until the reader asks for a new one: every call updates
# Agora's memory of what was shown, so a new call may reorder the list and a click on an
# article that left the page would be lost.
rec_key = f"rec-{user_id}-{feed_id}-{window}"
if refresh or rec_key not in st.session_state:
    params = {"user_id": user_id}
    if lam is not None:
        params["lam"] = lam
    try:
        if feed_id == "live":
            fetched = api("GET", "/recommend_live", params={**params, "limit": 26})
        else:
            fetched = api("GET", "/recommend", params={**params, "feed_id": feed_id})
    except requests.HTTPError as error:
        st.error(f"Agora could not rank this feed: {error.response.text}")
        st.stop()
    st.session_state[rec_key] = fetched
    st.session_state[f"served-{rec_key}"] = st.session_state.get(f"served-{rec_key}", 0) + 1
data = st.session_state[rec_key]
everything = data["items"]
items = everything[:top_k]
by_interest = sorted(everything, key=lambda x: -(x.get("relevance_score") or 0))
interest_rank = {x["article_id"]: i + 1 for i, x in enumerate(by_interest)}

# Log one 'display' event per shown article, once per served list and page size.
shown_key = f"shown-{rec_key}-{st.session_state[f'served-{rec_key}']}-{top_k}"
if shown_key not in st.session_state:
    for item in items:
        log_event("display", item, data)
    st.session_state[shown_key] = True

# Reader band: what this reader usually reads.
try:
    profile = api("GET", f"/users/{user_id}/profile")
except requests.HTTPError:
    profile = {"reads": 0, "topics": {}}
habits = Counter(profile.get("topics", {}))
favourites = [t for t, _ in habits.most_common(3)]
st.markdown(f'<div class="ag-reader"><h3>{esc(name)}{(", " + esc(habit)) if habit else ""}</h3>'
            f'<p class="sub">What {esc(name.lower() if name == "This reader" else name)} read during '
            f'the week of 23 November 2023 ({profile.get("reads", 0)} articles)</p>'
            f'{strip(habits)}</div>', unsafe_allow_html=True)


def reason(item) -> tuple[str, bool]:
    label = topic_of(item.get("topic"))[0]
    if item.get("topic") in protected and item["rank"] < interest_rank.get(item["article_id"], item["rank"]):
        return f"Brought up by Agora: {label.lower()} stays in view", True
    if item.get("topic") in favourites:
        return f"You often read {label.lower()}", False
    return "Close to articles you read", False


selection, compare, behind = st.tabs(["Your selection", "With and without Agora", "How it works"])

with selection:
    st.markdown(f'<p class="ag-caption">Topic mix of these {len(items)} articles</p>'
                f'{strip(Counter(x.get("topic") for x in items))}', unsafe_allow_html=True)
    st.write("")
    for item in items:
        label, colour = topic_of(item.get("topic"))
        why, boosted = reason(item)
        image = item.get("image")
        thumb = (f'<div class="ag-thumb" style="background-image:url(\'{esc(image)}\')"></div>'
                 if image else "")
        left, right = st.columns([11, 2], vertical_alignment="center")
        with left:
            st.markdown(
                f'<div class="ag-item{"" if image else " noimg"}" style="--c:{colour}">'
                f'<div class="ag-bar"></div>{thumb}<div class="ag-body">'
                f'<div class="ag-top"><span class="ag-num">{item["rank"]}</span><b>{esc(label)}</b>, '
                f'{esc(item.get("outlet"))}, {esc(when(item.get("published_at")))}</div>'
                f'<div class="ag-title">{esc(item["title"])}</div>'
                f'<div class="ag-lead">{esc(item.get("lead"))}</div>'
                f'<span class="ag-why{" boost" if boosted else ""}">{esc(why)}</span>'
                f'</div></div>', unsafe_allow_html=True)
        with right:
            if st.button("Read", key=f"read-{rec_key}-{item['article_id']}", width="stretch"):
                log_event("click", item, data)
                st.toast(f"Saved: {name} read “{item['title'][:60]}”")
            if item.get("url"):
                st.link_button("Open", item["url"], width="stretch")

with compare:
    plain = by_interest[:top_k]
    in_plain = {x["article_id"] for x in plain}
    count = lambda rows: sum(x.get("topic") in protected for x in rows)
    names = " and ".join(sorted(topic_of(t)[0].lower() for t in protected)) or "protected topics"
    a, b = st.columns(2, gap="large")
    with a:
        st.subheader("Your interests only")
        st.markdown(f'<div class="ag-count">{count(plain)}<small>{names} articles in the top {top_k}</small></div>'
                    f'{strip(Counter(x.get("topic") for x in plain))}', unsafe_allow_html=True)
        st.write("")
        for i, x in enumerate(plain, 1):
            st.markdown(f'<div class="ag-row" style="--c:{topic_of(x.get("topic"))[1]}"><span class="n">{i}</span>'
                        f'<span class="t">{esc(x["title"])}</span></div>', unsafe_allow_html=True)
    with b:
        st.subheader("With Agora")
        st.markdown(f'<div class="ag-count">{count(items)}<small>{names} articles in the top {top_k}</small></div>'
                    f'{strip(Counter(x.get("topic") for x in items))}', unsafe_allow_html=True)
        st.write("")
        for x in items:
            new = x["article_id"] not in in_plain
            st.markdown(f'<div class="ag-row{" new" if new else ""}" style="--c:{topic_of(x.get("topic"))[1]}">'
                        f'<span class="n">{x["rank"]}</span><span class="t">{esc(x["title"])}</span>'
                        f'{"<span class=tag>added by Agora</span>" if new else ""}</div>', unsafe_allow_html=True)
    if window == "Only my interests":
        st.info("The window is set to Only my interests, so both lists are the same. Choose Balanced or Wide.")

with behind:
    st.markdown(
        "1. **Your habits.** Agora learns from the articles you read: topics, outlets and the words they use.\n"
        "2. **A shortlist.** From the day's articles it keeps 26: the closest to your habits, with at least one "
        "article of each topic.\n"
        "3. **Your ranking.** A model trained on what real readers of these outlets clicked scores how likely you are to read each one.\n"
        "4. **The public square.** Agora remembers what it showed recently. When politics or environment news "
        "has been missing, it moves the most relevant of those stories up.\n"
        "5. **Your clicks.** Every article shown and every article read is saved, to measure and improve the "
        "selection.")
    latency = data.get("latency_ms")
    st.caption(f"{len(everything)} articles ranked"
               + (f" in {latency:.0f} ms" if latency is not None else "")
               + f". Model `{data['model_version']}`, window {window.lower()}.")
    st.dataframe(api("GET", "/interactions", params={"user_id": user_id, "limit": 10}),
                 width="stretch", hide_index=True)
