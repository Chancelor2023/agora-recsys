# frontend
Owner: Mathis.

Streamlit reader interface: `streamlit run src/agora/frontend/app.py` (API address: `AGORA_API_URL`, default http://127.0.0.1:8000).

Choose a reader profile and a daily feed; the page calls `/recommend` and shows the reranked articles with outlet, topic and lead. Each shown article is logged once as a `display` event, and "Read" logs a `click`; the last stored events are listed at the bottom of the page.
