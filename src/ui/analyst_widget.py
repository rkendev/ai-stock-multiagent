# src/ui/analyst_widget.py
from __future__ import annotations

import os
import streamlit as st
from analyst.compose import build_facts, render_analyst_take, log_qa


def render(
    ticker: str,
    data_dir: str = "data",
    out_dir: str = "output",
    sent_root: str = "output/sentiment",
) -> None:
    """Render the 'Ask the Analyst' widget."""
    st.subheader("Ask the Analyst (beta)")

    # Determine mode (LLM vs offline) from env
    llm_enabled = (
        os.getenv("ANALYST_ENABLED", "1") not in ("0", "false", "False")
        and bool(os.getenv("OPENAI_API_KEY"))
    )
    mode = "LLM" if llm_enabled else "offline"
    st.caption("Mode: **{}** (uses only local facts; no live web)".format(mode))

    # Prompt
    q = st.text_area(
        "Question about this ticker",
        placeholder="e.g., What's driving the recent move?",
        key=f"analyst_q_{ticker}",
    )

    # Action
    ask = st.button("Ask", key=f"ask_btn_{ticker}")

    if ask and q.strip():
        with st.spinner("Thinking..."):
            facts = build_facts(ticker, data_dir=data_dir, out_dir=out_dir, sent_root=sent_root)
            facts_q = {**facts, "question": q.strip()}
            ans = render_analyst_take(facts_q)

        # Display the answer
        st.markdown(ans or "_(No answer — check API key or environment configuration.)_")

        # Append Q/A to output/<TICKER>/analyst_qa.jsonl (fail-soft)
        try:
            log_qa(
                out_dir=out_dir,
                ticker=ticker,
                record={
                    "mode": mode,
                    "question": q.strip(),
                    "answer": ans or "",
                },
            )
        except Exception:
            # Never break the UI if local logging fails
            pass
