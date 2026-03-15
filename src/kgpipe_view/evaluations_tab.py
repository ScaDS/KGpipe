from __future__ import annotations

import streamlit as st

try:
    from kgpipe_view.meta_kg_query import query_kg_data
except ModuleNotFoundError:
    from meta_kg_query import query_kg_data


def render_evaluations_tab(endpoint_url: str) -> None:
    st.subheader("Evaluations")
    st.caption("Shows evaluation relations under `kgp:Evaluation`, including standalone evaluation nodes.")

    if st.button("Load evaluation hierarchy"):
        try:
            evaluation_hierarchy_df = query_kg_data(endpoint_url)
        except Exception as exc:  # pragma: no cover - network dependent path
            st.error(f"Could not query `{endpoint_url}`: {exc}")
        else:
            if evaluation_hierarchy_df.empty:
                st.info("No `kgp:Evaluation` subclass hierarchy returned by the endpoint.")
            else:
                st.dataframe(evaluation_hierarchy_df, use_container_width=True)
