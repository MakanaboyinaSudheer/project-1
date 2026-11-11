from discovery_agent.ui.components import render_parallel_hypotheses


def test_ui_component_import():
    """Simple test to ensure the Streamlit UI module imports successfully."""
    assert callable(render_parallel_hypotheses)
