from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from kgpipe_view.diagram_tab import render_diagram_tab
    from kgpipe_view.evaluations_tab import render_evaluations_tab
    from kgpipe_view.pipelines_tab import render_pipelines_tab
    from kgpipe_view.tasks_tab import render_tasks_tab
except ModuleNotFoundError:
    # Support direct script execution via: streamlit run src/kgpipe_view/kgpipe_view.py
    import importlib

    render_diagram_tab = importlib.import_module("diagram_tab").render_diagram_tab
    render_evaluations_tab = importlib.import_module("evaluations_tab").render_evaluations_tab
    render_pipelines_tab = importlib.import_module("pipelines_tab").render_pipelines_tab
    render_tasks_tab = importlib.import_module("tasks_tab").render_tasks_tab


st.set_page_config(page_title="KGpipe View", layout="wide")
st.title("KGpipe View")
st.caption("Explore the KGpipe meta knowledge graph rendered from Owl/Turtle.")

base_dir = Path(__file__).resolve().parent
ttl_path = base_dir / "kgpipe.owl.ttl"
mermaid_path = base_dir / "kgpipe.owl.mmd"

diagram_tab, tasks_tab, pipelines_tab, evaluations_tab = st.tabs(["Ontology Diagram", "Tasks", "Pipelines", "Evaluations"])

with diagram_tab:
    render_diagram_tab(ttl_path=ttl_path, mermaid_path=mermaid_path)

with tasks_tab:
    endpoint_url = render_tasks_tab()

with pipelines_tab:
    render_pipelines_tab(endpoint_url=endpoint_url)

with evaluations_tab:
    render_evaluations_tab(endpoint_url=endpoint_url)