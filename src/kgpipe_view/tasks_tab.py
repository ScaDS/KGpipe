from __future__ import annotations

import streamlit as st

try:
    from kgpipe_view.meta_kg_query import query_task_hierarchy, query_tasks_implementations
except ModuleNotFoundError:
    from meta_kg_query import query_task_hierarchy, query_tasks_implementations


def render_tasks_tab() -> str:
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

    return endpoint_url
