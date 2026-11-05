"""Streamlit frontend for the Discovery Agent."""

import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from discovery_agent.graph.workflow import get_run_status, run_discovery

st.set_page_config(page_title="Discovery Agent", page_icon="🔬", layout="wide")

st.title("AI Discovery Agent Dashboard")
st.markdown("Multi-agent workflow powered by LangGraph & SQLite Checkpoints.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "is_running" not in st.session_state:
    st.session_state.is_running = False

with st.sidebar:
    st.header("Experiment Setup")
    topic_input = st.text_area(
        "Research Topic", placeholder="e.g., Federated learning over multimodal clinical data"
    )

    if st.button("Run Workflow", type="primary", use_container_width=True):
        if topic_input.strip():
            st.session_state.is_running = True
            st.session_state.topic = topic_input
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.pop("final_state", None)
        else:
            st.error("Please enter a topic.")

if st.session_state.is_running:
    st.info(f"**Active Thread ID:** `{st.session_state.thread_id}`")

    with st.status("Executing LangGraph Pipeline...", expanded=True) as status:
        st.write("Cycling through Review, Hypothesis, Code, Critic, and Execution nodes...")
        try:
            final_state = run_discovery(topic=st.session_state.topic, thread_id=st.session_state.thread_id)
            st.session_state.final_state = final_state
            status.update(label="Workflow completed successfully!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Pipeline Failed", state="error")
            st.error(f"Execution Error: {str(e)}")
        finally:
            st.session_state.is_running = False

if "final_state" in st.session_state:
    state = st.session_state.final_state

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Research Output")
        st.write(f"**Topic:** {state.get('research_topic', 'N/A')}")

        if state.get("memory_summary"):
            st.markdown("### Agent Memory Summary")
            st.info(state["memory_summary"])

        code_attempts = state.get("code_attempts", [])
        if code_attempts:
            st.markdown(f"### Code Execution Attempts ({len(code_attempts)})")
            for idx, attempt in enumerate(code_attempts):
                with st.expander(f"Attempt {idx + 1}"):
                    st.json(attempt)

    with col2:
        st.subheader("SQLite Checkpoint State")
        try:
            raw_status = get_run_status(st.session_state.thread_id)
            if raw_status:
                st.json(raw_status)
            else:
                st.warning("No checkpoint data found in volumes/checkpoints.sqlite")
        except Exception as e:
            st.error(f"Could not load checkpoint status: {str(e)}")
