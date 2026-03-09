from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from streamlit_elements import elements, dashboard, mui

from meta_kg_query import query_task_hierarchy, query_tasks_implementations, query_pipeline_hierarchy, query_evaluation_hierarchy, query_kg_data
from owl_to_mermaid import convert_and_write_mermaid, get_available_layers


def _render_mermaid(mermaid_text: str, height: int = 900) -> None:
    """Render Mermaid source in Streamlit using Mermaid JS."""
    mermaid_json = json.dumps(mermaid_text)
    html = f"""
    <style>
      .mermaid-wrapper {{
        background: #ffffff;
        border: 1px solid #d9d9d9;
        border-radius: 8px;
        padding: 16px;
      }}
      .mermaid-wrapper .mermaid {{
        color: #111111;
      }}
    </style>
    <div class="mermaid-wrapper">
      <div class="mermaid"></div>
    </div>
    <script type="module">
      import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
      mermaid.initialize({{ startOnLoad: false, securityLevel: "loose" }});
      const element = document.querySelector(".mermaid");
      element.textContent = {mermaid_json};
      await mermaid.run({{ nodes: [element] }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


st.set_page_config(page_title="KGpipe View", layout="wide")
st.title("KGpipe View")
st.caption("Explore the KGpipe meta knowledge graph rendered from Owl/Turtle.")

base_dir = Path(__file__).resolve().parent
ttl_path = base_dir / "kgpipe.owl.ttl"
mermaid_path = base_dir / "kgpipe.owl.mmd"

diagram_tab, tasks_tab, pipelines_tab, evaluations_tab = st.tabs(["Ontology Diagram", "Tasks", "Pipelines", "Evaluations"])

with diagram_tab:
    try:
        layer_options = get_available_layers(ttl_path)
        selected_layers = st.multiselect(
            "Layers",
            options=layer_options,
            default=layer_options,
            key="layer-filter",
        )
        mermaid_code = convert_and_write_mermaid(
            ttl_path=ttl_path,
            output_path=mermaid_path,
            layer_filter=selected_layers,
        )
    except Exception as exc:  # pragma: no cover - UI fallback path
        st.error(f"Failed to convert `{ttl_path.name}` to Mermaid: {exc}")
    else:
        st.success(
            f"Generated Mermaid from `{ttl_path.name}` and saved `{mermaid_path.name}`."
        )
        if selected_layers:
            st.caption(f"Current layer filter: `{', '.join(selected_layers)}`")
        else:
            st.caption("Current layer filter: `none`")
        _render_mermaid(mermaid_code)
        with st.expander("Show Mermaid source"):
            st.code(mermaid_code, language="mermaid")

with tasks_tab:
    endpoint_url = st.text_input(
        "Meta KG SPARQL endpoint",
        value="http://localhost:8890/sparql",
        help="SPARQL endpoint for the live meta knowledge graph.",
    )
    if st.button("Load task implementations", type="primary"):
        try:
            task_implementation_df = query_tasks_implementations(endpoint_url)
        except Exception as exc:  # pragma: no cover - network dependent path
            st.error(f"Could not query `{endpoint_url}`: {exc}")
        else:
            if task_implementation_df.empty:
                st.info("No task-implementation mappings returned by the endpoint.")
            else:
                st.dataframe(task_implementation_df, use_container_width=True)

    st.divider()
    st.subheader("Task hierarchy")
    st.caption("Shows subclass relations under `kgp:Task`, including standalone task nodes.")

    if st.button("Load task hierarchy"):
        try:
            task_hierarchy_df = query_task_hierarchy(endpoint_url)
        except Exception as exc:  # pragma: no cover - network dependent path
            st.error(f"Could not query `{endpoint_url}`: {exc}")
        else:
            if task_hierarchy_df.empty:
                st.info("No `kgp:Task` subclass hierarchy returned by the endpoint.")
            else:
                st.dataframe(task_hierarchy_df, use_container_width=True)

with pipelines_tab:
    st.subheader("Pipelines")
    st.caption("Shows pipeline relations under `kgp:Pipeline`, including standalone pipeline nodes.")

    if st.button("Load pipeline hierarchy"):
        try:
            pipeline_hierarchy_df = query_pipeline_hierarchy(endpoint_url)
        except Exception as exc:  # pragma: no cover - network dependent path
            st.error(f"Could not query `{endpoint_url}`: {exc}")
        else:
            if pipeline_hierarchy_df.empty:
                st.info("No `kgp:Pipeline` subclass hierarchy returned by the endpoint.")
            else:
                st.dataframe(pipeline_hierarchy_df, use_container_width=True)

    with elements("dashboard"):
        layout = [
            dashboard.Item("item1", 0, 0, 2, 2),
            dashboard.Item("item2", 2, 0, 2, 2)
        ]
        
        with dashboard.Grid(layout):
            mui.Paper("Draggable Panel 1", key="item1")
            mui.Paper("Draggable Panel 2", key="item2")

with evaluations_tab:
    st.subheader("Evaluations")
    st.caption("Shows evaluation relations under `kgp:Evaluation`, including standalone evaluation nodes.")

    if st.button("Load evaluation hierarchy"):
        try:
            evaluation_hierarchy_df = query_kg_data(endpoint_url) #query_evaluation_hierarchy(endpoint_url)
        except Exception as exc:  # pragma: no cover - network dependent path
            st.error(f"Could not query `{endpoint_url}`: {exc}")
        else:
            if evaluation_hierarchy_df.empty:
                st.info("No `kgp:Evaluation` subclass hierarchy returned by the endpoint.")
            else:
                st.dataframe(evaluation_hierarchy_df, use_container_width=True)