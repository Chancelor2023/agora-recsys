"""Agora reader interface (Streamlit).

Run the API first, then:  streamlit run src/agora/frontend/app.py
The API address can be changed with AGORA_API_URL (default http://127.0.0.1:8000).
"""
import os
import uuid

import requests
import streamlit as st

API_URL = os.environ.get("AGORA_API_URL", "http://127.0.0.1:8000").rstrip("/")
TOPIC_LABELS = {
    "politics": "Politics", "world": "World", "uk_news": "UK news", "environment": "Environment",
    "business": "Business", "sport": "Sport", "culture": "Culture", "lifestyle": "Lifestyle",
    "tech_science": "Tech & science", "health": "Health", "crime": "Crime", "opinion": "Opinion",
}

st.set_page_config(page_title="Agora", page_icon="📰", layout="wide")


def api(method: str, path: str, **kwargs):
    response = requests.request(method, API_URL + path, timeout=10, **kwargs)
    response.raise_for_status()
    return response.json()


def log_event(event_type: str, item: dict, data: dict) -> dict:
    return api("POST", "/log_interaction", json={
        "event_id": str(uuid.uuid4()), "user_id": data["user_id"], "feed_id": data["feed_id"],
        "article_id": item["article_id"], "event_type": event_type,
        "displayed_rank": item["rank"], "model_version": data["model_version"]})


st.title("Agora")
st.caption("Personalised news, reranked so that public-affairs news keeps a fair share of attention.")

try:
    mode = api("GET", "/mode")
except requests.RequestException as error:
    st.error(f"The API is not reachable at {API_URL}. Start it with "
             f"`python -m uvicorn agora.api.main:app` ({error.__class__.__name__}).")
    st.stop()

with st.sidebar:
    st.subheader("Reader")
    if mode["is_mock"]:
        st.warning("Mock mode: run the data pipeline and train the model to serve real feeds.")
        user_id, feed_id = "demo-user", "demo-feed"
    else:
        users = api("GET", "/users")
        user_id = st.selectbox("Reader profile", users)
        live = api("GET", "/live/status")
        source = st.radio("News", ["Today's news (live RSS)", "IDEA daily feed (2023)"]
                          if live.get("available") else ["IDEA daily feed (2023)"])
        if source.startswith("Today"):
            feed_id = "live"
            st.caption(f"{live['articles']} articles fetched {str(live.get('fetched_at'))[:16].replace('T', ' ')} UTC")
        else:
            feeds = api("GET", f"/users/{user_id}/feeds")
            labels = {f["feed_id"]: f"{f['generated_at'][:16].replace('T', ' ')} ({f['split']})" for f in feeds}
            # key per reader: otherwise Streamlit keeps the previous reader's feed selected
            feed_id = st.selectbox("Daily feed", list(labels), format_func=labels.get,
                                   key=f"feed-{user_id}")
        if not live.get("available"):
            st.caption(f"Live news off: {live.get('reason', '')}")
    top_k = st.slider("Articles shown", 5, 26, 10)
    refresh = st.button("New recommendation")
    st.caption(f"Model: `{mode['model_version']}`")

# Keep the list that was served until the reader asks for a new one: every call to the
# API updates SCRUF-D's exposure state, so a new call may reorder the list, and a click
# on an article that left the page would be lost.
rec_key = f"rec-{user_id}-{feed_id}"
if refresh or rec_key not in st.session_state:
    try:
        if feed_id == "live":
            fetched = api("GET", "/recommend_live", params={"user_id": user_id, "limit": 26})
        else:
            fetched = api("GET", "/recommend", params={"user_id": user_id, "feed_id": feed_id})
    except requests.HTTPError as error:
        st.error(f"The API refused this request: {error.response.text}")
        st.stop()
    st.session_state[rec_key] = fetched
    st.session_state[f"served-{rec_key}"] = st.session_state.get(f"served-{rec_key}", 0) + 1
data = st.session_state[rec_key]
items = data["items"][:top_k]

latency = data.get("latency_ms")
st.caption(f"{len(data['items'])} articles ranked"
           + (f" in {latency:.0f} ms" if latency is not None else "")
           + (" · mock data" if data["is_mock"] else ""))

# Log one 'display' event per shown article, once per served list and page size.
shown_key = f"shown-{rec_key}-{st.session_state[f'served-{rec_key}']}-{top_k}"
if shown_key not in st.session_state:
    for item in items:
        log_event("display", item, data)
    st.session_state[shown_key] = True

for item in items:
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            topic = TOPIC_LABELS.get(item.get("topic") or "", item.get("topic") or "")
            st.markdown(f"**{item['rank']}. {item['title']}**")
            st.caption(f"{item['outlet']} · {topic} · {item['published_at'][:10]}")
            if item.get("lead"):
                st.write(item["lead"])
        with right:
            if st.button("Read", key=f"read-{feed_id}-{item['article_id']}"):
                result = log_event("click", item, data)
                st.success(f"Saved ({result['total_events']} events)")
            if item.get("url"):
                st.link_button("Open", item["url"])

with st.expander("Last interactions stored in the database"):
    st.dataframe(api("GET", "/interactions", params={"user_id": user_id, "limit": 10}))
