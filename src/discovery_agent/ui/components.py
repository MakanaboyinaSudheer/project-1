"""Streamlit UI components for the Discovery Agent dashboard."""

from typing import Any

import streamlit as st


def render_parallel_hypotheses(hypotheses: list[dict[str, Any]]):
    """Renders multiple hypotheses side-by-side using Streamlit columns."""
    if not hypotheses:
        st.info("No parallel hypotheses generated yet.")
        return

    st.subheader("Parallel Research Angles")

    # Create dynamic columns based on the number of parallel outputs
    cols = st.columns(len(hypotheses))

    for idx, (col, hyp) in enumerate(zip(cols, hypotheses, strict=False)):
        with col:
            angle = hyp.get("angle", f"Angle {idx+1}").title()
            st.markdown(f"### {angle}")

            # Use an expander to keep the UI clean if the content is long
            with st.expander("View Details", expanded=True):
                st.write(hyp.get("content", "No content available."))

            # Add a selection button for human-in-the-loop override
            if st.button(f"Promote {angle}", key=f"promote_{angle}"):
                st.session_state["selected_angle"] = angle
                st.success(f"{angle} angle promoted to execution phase!")
