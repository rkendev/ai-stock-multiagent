from __future__ import annotations
import os
import streamlit as st
from analyst.compose import build_facts, render_analyst_take


def render(ticker: str, data_dir: str = "data", out_dir: str = "output", sent_root: str = "output/sentiment"):
    st.subheader("Ask the Analyst (beta)")
    q = st.text_area("Question about this ticker", placeholder="e.g., What's driving the recent move?")
    col1, col2 = st.columns([1, 4])
    with col1:
        go = st.button("Ask", key="ask_btn", width="stretch")
    if go and q.strip():
        with st.spinner("Thinking..."):
            facts = build_facts(ticker, data_dir=data_dir, out_dir=out_dir, sent_root=sent_root)
            # lightweight prompt reuse: append the question to facts
            facts_q = dict(facts)
            facts_q["question"] = q.strip()
            ans = render_analyst_take(facts_q)
        st.markdown(ans or "_(No answer — check API key or environment configuration.)_")

    mode = (
        "LLM"
        if (os.getenv("ANALYST_ENABLED", "1") not in ("0", "false", "False") and os.getenv("OPENAI_API_KEY"))
        else "offline"
    )
    st.caption(f"Mode: **{mode}** (uses only local facts; no live web)")
