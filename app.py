"""Streamlit UI for the Music Recommender Agent."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.agent import AgentResult, MusicAgent
from src.recommender import load_songs
from pathlib import Path
import pandas as pd
import os

REPO_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def get_agent():
    songs = load_songs(str(REPO_ROOT / "data" / "songs.csv"))
    return MusicAgent(songs=songs)


st.set_page_config(
    page_title="Music Recommender Agent",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Music Recommender Agent")
st.caption(
    "Type a natural-language music request and the AI agent will plan, "
    "retrieve context, call the recommender, and explain its picks."
)

has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
if not has_key:
    st.warning(
        "ANTHROPIC_API_KEY not set — the agent will use deterministic fallback mode. "
        "Add your key to `.env` and restart the app for full LLM-driven recommendations."
    )

with st.sidebar:
    st.header("Settings")
    show_trace = st.toggle("Show agent trace", value=False)
    show_raw = st.toggle("Show raw scores breakdown", value=False)
    st.divider()
    st.markdown("**Example queries:**")
    examples = [
        "chill lofi for studying late at night",
        "pump-up EDM for a hard workout",
        "relaxing jazz for a coffee shop",
        "hidden gems, nothing on the charts",
        "I'm feeling melancholy — something bittersweet",
        "high energy pop for a road trip",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", width="stretch"):
            st.session_state["query_input"] = ex

query = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g. chill lofi for studying late at night",
    key="query_input",
)

if st.button("🎧 Get Recommendations", type="primary", width="stretch") and query.strip():
    agent = get_agent()
    with st.spinner("Agent is thinking..."):
        result = agent.run(query.strip())

    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    result: AgentResult = st.session_state["last_result"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{result.evaluation.confidence:.0%}")
    col2.metric("Mode", result.mode)
    col3.metric("RAG Used", "Yes" if result.retrieval_used else "No")
    col4.metric("Fallback", "Yes" if result.fallback_used else "No")

    st.markdown(f"**{result.summary}**")

    if result.recommendations:
        rows = []
        for i, rec in enumerate(result.recommendations, 1):
            rows.append({
                "#": i,
                "Title": rec.get("title", ""),
                "Artist": rec.get("artist", ""),
                "Genre": rec.get("genre", ""),
                "Mood": rec.get("mood", ""),
                "Energy": rec.get("energy", ""),
                "Popularity": rec.get("popularity", ""),
                "Score": f"{rec.get('score', 0):.2f}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Score": st.column_config.TextColumn(width="small"),
                "Energy": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Popularity": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%d"
                ),
            },
        )

    if result.evaluation.issues:
        with st.expander("⚠️ Evaluator issues"):
            for issue in result.evaluation.issues:
                st.markdown(f"- {issue}")

    if show_raw and result.recommendations:
        with st.expander("Score breakdown per song"):
            for rec in result.recommendations:
                st.markdown(f"**{rec.get('title')}** ({rec.get('score', 0):.2f})")
                st.text(rec.get("why", ""))

    if show_trace:
        with st.expander("Agent trace", expanded=True):
            for step in result.trace:
                action = step.action
                detail = step.detail
                if action == "user_query":
                    st.markdown(f"**[{step.step}] Query:** `{detail.get('query', '')}`")
                elif action.startswith("tool:"):
                    tool_name = action.replace("tool:", "")
                    summary = detail.get("result_summary", {})
                    st.markdown(f"**[{step.step}] Tool call:** `{tool_name}`")
                    st.json(summary)
                elif action == "evaluate":
                    st.markdown(f"**[{step.step}] Self-check:**")
                    st.json({
                        "confidence": detail.get("confidence"),
                        "ok": detail.get("ok"),
                        "issues": detail.get("issues"),
                        "breakdown": detail.get("breakdown"),
                    })
                elif action == "error":
                    st.error(f"[{step.step}] {detail.get('message', '')}")
                elif action == "fallback":
                    st.warning(f"[{step.step}] Fallback: {detail.get('reason', '')}")
                elif action == "assistant_text":
                    st.markdown(f"**[{step.step}] Agent:** {detail.get('text', '')}")
                elif action == "retry":
                    st.info(f"[{step.step}] Retry: {detail.get('reason', '')} → {detail.get('new_mode', '')}")

    if result.error:
        with st.expander("Error details"):
            st.code(result.error)
