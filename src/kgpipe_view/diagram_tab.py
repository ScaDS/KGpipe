from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    from kgpipe_view.owl_to_mermaid import convert_and_write_mermaid, get_available_layers
    from kgpipe_view.ui_common import render_mermaid
except ModuleNotFoundError:
    from owl_to_mermaid import convert_and_write_mermaid, get_available_layers
    from ui_common import render_mermaid


def render_diagram_tab(ttl_path: Path, mermaid_path: Path) -> None:
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
        return

    st.success(
        f"Generated Mermaid from `{ttl_path.name}` and saved `{mermaid_path.name}`."
    )
    if selected_layers:
        st.caption(f"Current layer filter: `{', '.join(selected_layers)}`")
    else:
        st.caption("Current layer filter: `none`")
    render_mermaid(mermaid_code)
    with st.expander("Show Mermaid source"):
        st.code(mermaid_code, language="mermaid")
