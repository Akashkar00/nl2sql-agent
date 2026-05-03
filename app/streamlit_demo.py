"""
Streamlit demo for NL2SQL agent.

Run:
    streamlit run app/streamlit_demo.py

Hits the FastAPI backend (configurable via API_URL env var).
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NL2SQL Agent", layout="wide")
st.title("NL2SQL Agent")
st.caption("Ask in English. Watch the model generate, execute, and self-correct.")

# ---- Sidebar: DB picker + config ----
with st.sidebar:
    st.header("Config")
    try:
        dbs = requests.get(f"{API_URL}/databases", timeout=5).json().get("databases", [])
    except Exception as e:
        st.error(f"backend unreachable: {e}")
        dbs = []

    db_id = st.selectbox("database", dbs) if dbs else st.text_input("db_id")
    use_agent = st.toggle("self-correction loop", value=True,
                          help="when off, runs single-shot baseline")
    max_retries = st.slider("max retries", 0, 5, 2, disabled=not use_agent)
    evidence = st.text_area("evidence (optional, BIRD-style hint)", "")

# ---- Main: question input ----
question = st.text_input("Your question", placeholder="How many customers are from Germany?")

if st.button("Run", type="primary", disabled=not (question and db_id)):
    with st.spinner("generating + executing..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "db_id": db_id,
                    "evidence": evidence,
                    "use_agent": use_agent,
                    "max_retries": max_retries,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"request failed: {e}")
            st.stop()

    # ---- Render ----
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Generated SQL")
        st.code(data["sql"], language="sql")

        if data["success"]:
            st.subheader("Result")
            if data["rows"]:
                st.dataframe(
                    {col: [r[i] for r in data["rows"]] for i, col in enumerate(data["columns"] or [])}
                )
            else:
                st.info("query ran successfully but returned 0 rows")
        else:
            st.error(f"execution failed: {data['error']}")

    with col2:
        st.subheader("Agent trace")
        st.metric("retries used", data["retries_used"])
        for att in data["attempts"]:
            with st.expander(f"attempt {att['attempt']}: {att['status']}"):
                st.code(att["sql"], language="sql")
