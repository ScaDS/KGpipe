from __future__ import annotations

import json
from uuid import uuid4

import streamlit as st

from kgpipe.common.systemgraph import PipeKG
try:
    from kgpipe_view.meta_kg_query import query_pipeline_hierarchy
    from kgpipe_view.ui_common import render_mermaid
except ModuleNotFoundError:
    from meta_kg_query import query_pipeline_hierarchy
    from ui_common import render_mermaid


def _entity_to_task_label(entity) -> str:
    """Best-effort conversion from implementation-like object to task label."""
    name = getattr(entity, "name", "") or ""
    return _task_label_to_base_name(str(name))


def _task_label_to_base_name(label: str) -> str:
    base = str(label or "").strip()
    if base.endswith("Impl"):
        return base[:-4]
    return base


@st.cache_data(show_spinner=False)
def _get_task_io_specs() -> dict[str, dict[str, set[str]]]:
    return PipeKG().list_task_io_specs()


def _shared_formats(
    source_task_name: str,
    target_task_name: str,
    specs: dict[str, dict[str, set[str]]],
) -> set[str]:
    source_outputs = specs.get(_task_label_to_base_name(source_task_name), {}).get("outputs", set())
    target_inputs = specs.get(_task_label_to_base_name(target_task_name), {}).get("inputs", set())
    if not source_outputs or not target_inputs:
        return {"*"}
    return source_outputs.intersection(target_inputs)


def _edge_label(formats: set[str]) -> str:
    if not formats:
        return ""
    if formats == {"*"}:
        return "any"
    return ", ".join(sorted(formats))


def _task_io_summary(task_name: str, specs: dict[str, dict[str, set[str]]]) -> str:
    task_spec = specs.get(_task_label_to_base_name(task_name), {})
    inputs = sorted(task_spec.get("inputs", set()))
    outputs = sorted(task_spec.get("outputs", set()))
    in_text = ", ".join(inputs) if inputs else "-"
    out_text = ", ".join(outputs) if outputs else "-"
    return f"in: {in_text} | out: {out_text}"


def _task_options_from_implementations(implementations: list) -> list[str]:
    labels = {_entity_to_task_label(entity) for entity in implementations}
    return sorted(label for label in labels if label)


def _pipeline_to_mermaid(
    pipeline_nodes: list[dict[str, str]],
    pipeline_edges: list[dict[str, object]],
) -> str:
    if not pipeline_nodes:
        return "flowchart LR\n  empty[Empty pipeline]"

    def _safe_node_id(raw: str) -> str:
        return "n_" + "".join(ch if ch.isalnum() else "_" for ch in raw)

    node_ids = {node["id"] for node in pipeline_nodes}
    lines = ["flowchart LR"]
    for node in pipeline_nodes:
        node_id = _safe_node_id(node["id"])
        label = node["name"].replace(chr(34), chr(39))
        lines.append(f'{node_id}["{label}"]')

    for edge in pipeline_edges:
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if source not in node_ids or target not in node_ids or source == target:
            continue
        src = _safe_node_id(source)
        dst = _safe_node_id(target)
        formats = set(edge.get("formats", []))
        label = _edge_label(formats)
        if label:
            lines.append(f'{src} -->|{label}| {dst}')
        else:
            lines.append(f"{src} --> {dst}")
    return "\n".join(lines)


def _get_tasks() -> list:
    return PipeKG().list_taskImplementations()


def render_pipelines_tab(endpoint_url: str) -> None:
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

    st.divider()
    st.subheader("Pipeline builder")
    st.caption("Compose a DAG pipeline with reusable outputs and parallel branches.")

    if "pipeline_nodes" not in st.session_state:
        legacy_steps = st.session_state.get("pipeline_steps", [])
        migrated_nodes: list[dict[str, str]] = []
        if legacy_steps and isinstance(legacy_steps[0], str):
            migrated_nodes = [
                {"id": f"task-{uuid4().hex[:8]}", "name": task_name}
                for task_name in legacy_steps
            ]
        elif legacy_steps:
            migrated_nodes = legacy_steps
        st.session_state.pipeline_nodes = migrated_nodes
        st.session_state.pipeline_edges = []
        for idx in range(len(migrated_nodes) - 1):
            st.session_state.pipeline_edges.append(
                {
                    "from": migrated_nodes[idx]["id"],
                    "to": migrated_nodes[idx + 1]["id"],
                    "formats": [],
                }
            )
        st.session_state.pop("pipeline_steps", None)
    if "pipeline_edges" not in st.session_state:
        st.session_state.pipeline_edges = []

    implementations = _get_tasks()
    task_options = _task_options_from_implementations(implementations)
    task_specs = _get_task_io_specs()

    pipeline_nodes = st.session_state.pipeline_nodes
    pipeline_edges = st.session_state.pipeline_edges

    def _node_label(node: dict[str, str]) -> str:
        return f'{node["name"]} ({node["id"][-6:]})'

    if not task_options:
        st.info("No task implementations found in PipeKG yet.")
    else:
        task_option_labels = {
            task_name: f"{task_name}  [{_task_io_summary(task_name, task_specs)}]"
            for task_name in task_options
        }
        selected_task_name = st.selectbox(
            "Task to add",
            options=task_options,
            key="pipeline_builder_selected_task",
            format_func=lambda task_name: task_option_labels.get(task_name, task_name),
        )
        st.caption(f"Selected task spec: `{_task_io_summary(selected_task_name, task_specs)}`")

        selected_source_id = None
        if pipeline_nodes:
            source_options = [{"label": "No dependency (new branch/root)", "id": None}]
            for node in pipeline_nodes:
                formats = _shared_formats(node["name"], selected_task_name, task_specs)
                if formats:
                    source_options.append(
                        {
                            "label": f'{_node_label(node)}  [{_edge_label(formats)}]',
                            "id": node["id"],
                        }
                    )

            source_label = st.selectbox(
                "Connect new task from",
                options=[opt["label"] for opt in source_options],
                key="pipeline_builder_selected_source",
                help="Pick an upstream task output to reuse, or create a root branch with no dependency.",
            )
            source_lookup = {opt["label"]: opt["id"] for opt in source_options}
            selected_source_id = source_lookup[source_label]
        else:
            st.caption("First task creates the first root in the pipeline.")

        add_col, remove_col, clear_col = st.columns(3)
        with add_col:
            if st.button("Add task node", key="pipeline_builder_add_task"):
                new_id = f"task-{uuid4().hex[:8]}"
                st.session_state.pipeline_nodes.append({"id": new_id, "name": selected_task_name})
                if selected_source_id is not None:
                    source_node = next(
                        (node for node in st.session_state.pipeline_nodes if node["id"] == selected_source_id),
                        None,
                    )
                    formats = set()
                    if source_node is not None:
                        formats = _shared_formats(source_node["name"], selected_task_name, task_specs)
                    st.session_state.pipeline_edges.append(
                        {
                            "from": selected_source_id,
                            "to": new_id,
                            "formats": sorted(formats),
                        }
                    )
                st.rerun()
        with remove_col:
            if st.button("Remove last task node", key="pipeline_builder_remove_last"):
                if st.session_state.pipeline_nodes:
                    removed = st.session_state.pipeline_nodes.pop()
                    removed_id = removed["id"]
                    st.session_state.pipeline_edges = [
                        edge
                        for edge in st.session_state.pipeline_edges
                        if edge.get("from") != removed_id and edge.get("to") != removed_id
                    ]
                    st.rerun()
        with clear_col:
            if st.button("Clear pipeline", key="pipeline_builder_clear"):
                st.session_state.pipeline_nodes = []
                st.session_state.pipeline_edges = []
                st.session_state.pop("pipeline_builder_selected_task", None)
                st.session_state.pop("pipeline_builder_selected_source", None)
                st.rerun()

        if len(st.session_state.pipeline_nodes) >= 2:
            st.markdown("**Connect existing tasks**")
            id_to_node = {node["id"]: node for node in st.session_state.pipeline_nodes}
            source_node_id = st.selectbox(
                "From task",
                options=[node["id"] for node in st.session_state.pipeline_nodes],
                format_func=lambda node_id: _node_label(id_to_node[node_id]),
                key="pipeline_builder_connect_source",
            )
            existing_pairs = {
                (edge.get("from"), edge.get("to")) for edge in st.session_state.pipeline_edges
            }
            target_candidates: list[tuple[str, str]] = []
            for node in st.session_state.pipeline_nodes:
                if node["id"] == source_node_id:
                    continue
                if (source_node_id, node["id"]) in existing_pairs:
                    continue
                source_node = id_to_node[source_node_id]
                formats = _shared_formats(source_node["name"], node["name"], task_specs)
                if formats:
                    target_candidates.append(
                        (node["id"], f'{_node_label(node)}  [{_edge_label(formats)}]')
                    )

            if target_candidates:
                target_label = st.selectbox(
                    "To task",
                    options=[label for _, label in target_candidates],
                    key="pipeline_builder_connect_target",
                )
                target_lookup = {label: node_id for node_id, label in target_candidates}
                target_node_id = target_lookup[target_label]
                if st.button("Add dependency edge", key="pipeline_builder_add_edge"):
                    source_node = id_to_node[source_node_id]
                    target_node = id_to_node[target_node_id]
                    formats = _shared_formats(source_node["name"], target_node["name"], task_specs)
                    st.session_state.pipeline_edges.append(
                        {
                            "from": source_node_id,
                            "to": target_node_id,
                            "formats": sorted(formats),
                        }
                    )
                    st.rerun()
            else:
                st.caption("No additional compatible target task found for this source.")

    if pipeline_nodes:
        st.caption("Current pipeline")
        st.write(", ".join(_node_label(node) for node in pipeline_nodes))
        st.markdown("**Pipeline graph**")
        render_mermaid(_pipeline_to_mermaid(pipeline_nodes, pipeline_edges), height=300)

        incoming: dict[str, int] = {node["id"]: 0 for node in pipeline_nodes}
        outgoing: dict[str, int] = {node["id"]: 0 for node in pipeline_nodes}
        for edge in pipeline_edges:
            source = edge.get("from")
            target = edge.get("to")
            if source in outgoing:
                outgoing[source] += 1
            if target in incoming:
                incoming[target] += 1

        id_to_node = {node["id"]: node for node in pipeline_nodes}
        roots = [id_to_node[node_id] for node_id, count in incoming.items() if count == 0]
        leaves = [id_to_node[node_id] for node_id, count in outgoing.items() if count == 0]

        st.markdown("**Loose ends**")
        roots_text = ", ".join(_node_label(node) for node in roots) if roots else "none"
        leaves_text = ", ".join(_node_label(node) for node in leaves) if leaves else "none"
        st.caption(f"Open inputs (roots): {roots_text}")
        st.caption(f"Open outputs (leaves): {leaves_text}")

        st.code(
            json.dumps(
                {
                    "nodes": pipeline_nodes,
                    "edges": pipeline_edges,
                },
                indent=2,
            ),
            language="json",
        )
    else:
        st.info("Your pipeline is empty. Add one or more task nodes to start building.")
