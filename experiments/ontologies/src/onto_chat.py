"""Streamlit ontology chat prototype.

Run:
    uv run streamlit run experiments/ontologies/src/onto_chat.py
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import re
from textwrap import dedent

import streamlit as st
import streamlit.components.v1 as components
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL


EXAMPLE_OWL = dedent(
    """\
    @prefix ex: <http://example.org/> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .

    ex:Person a owl:Class .
    ex:Company a owl:Class .
    ex:Project a owl:Class .

    ex:worksFor a owl:ObjectProperty ;
        rdfs:domain ex:Person ;
        rdfs:range ex:Company .

    ex:worksOn a owl:ObjectProperty ;
        rdfs:domain ex:Person ;
        rdfs:range ex:Project .
    """
)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
KNOWN_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


@dataclass
class OntologySchema:
    classes: list[str]
    object_edges: list[tuple[str, str, str]]
    datatype_edges: list[tuple[str, str, str]]


def short_name(uri: URIRef) -> str:
    """Return a compact local name for URI nodes."""
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", maxsplit=1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return text


def parse_graph(raw_text: str, rdf_format: str) -> Graph:
    """Parse ontology text into an RDF graph."""
    graph = Graph()
    graph.parse(data=raw_text, format=rdf_format)
    return graph


def extract_schema(graph: Graph) -> OntologySchema:
    """Extract classes and property relations from graph."""
    classes: set[str] = set()
    object_edges: list[tuple[str, str, str]] = []
    datatype_edges: list[tuple[str, str, str]] = []

    for cls in graph.subjects(RDF.type, OWL.Class):
        if isinstance(cls, URIRef):
            classes.add(short_name(cls))
    for cls in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(cls, URIRef):
            classes.add(short_name(cls))

    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        if not isinstance(prop, URIRef):
            continue
        prop_name = short_name(prop)
        domains = [d for d in graph.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
        ranges = [r for r in graph.objects(prop, RDFS.range) if isinstance(r, URIRef)]
        for domain in domains or [URIRef("UnknownDomain")]:
            for rng in ranges or [URIRef("UnknownRange")]:
                src, dst = short_name(domain), short_name(rng)
                classes.update([src, dst])
                object_edges.append((src, prop_name, dst))

    for prop in graph.subjects(RDF.type, OWL.DatatypeProperty):
        if not isinstance(prop, URIRef):
            continue
        prop_name = short_name(prop)
        domains = [d for d in graph.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
        ranges = [r for r in graph.objects(prop, RDFS.range) if isinstance(r, URIRef)]
        for domain in domains or [URIRef("UnknownDomain")]:
            for rng in ranges or [URIRef("Literal")]:
                src, dst = short_name(domain), short_name(rng)
                classes.add(src)
                datatype_edges.append((src, prop_name, dst))

    return OntologySchema(
        classes=sorted(classes),
        object_edges=object_edges,
        datatype_edges=datatype_edges,
    )


def to_mermaid(schema: OntologySchema) -> str:
    """Serialize ontology schema as Mermaid classDiagram."""
    lines = ["classDiagram"]
    for cls_name in schema.classes:
        lines.append(f"  class {cls_name}")
    for src, rel, dst in schema.object_edges:
        lines.append(f"  {src} --> {dst} : {rel}")
    for src, rel, dst in schema.datatype_edges:
        lines.append(f"  {src} : {rel} -> {dst}")
    return "\n".join(lines)


def render_mermaid(mermaid_text: str) -> None:
    """Render Mermaid diagram in Streamlit via embedded HTML."""
    escaped = (
        mermaid_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = f"""
    <div class="mermaid">{escaped}</div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
      mermaid.initialize({{ startOnLoad: true, theme: "default" }});
    </script>
    """
    components.html(html, height=500, scrolling=True)


def draft_llm_prompt(user_request: str, ontology_text: str, rdf_format: str) -> str:
    """Build a prompt for a future LLM integration."""
    return dedent(
        f"""\
        You are editing an OWL ontology.

        Task:
        {user_request}

        Requirements:
        - Return only ontology text in {rdf_format} format.
        - Preserve existing prefixes when possible.
        - Declare all prefixes you use (especially xsd when using xsd:* datatypes).
        - Keep edits minimal and valid.
        - Do not include markdown fences.

        Current ontology:
        {ontology_text}
        """
    )


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences if model returns them."""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def extract_declared_prefixes(text: str) -> set[str]:
    """Extract declared prefixes from Turtle/N3 text."""
    return set(re.findall(r"@prefix\s+([A-Za-z][\w\-]*)\s*:", text))


def extract_used_prefixes(text: str) -> set[str]:
    """Extract prefixed terms used in Turtle/N3 text."""
    matches = re.findall(r"(?<!@prefix\s)([A-Za-z][\w\-]*)\:[A-Za-z_][\w\-]*", text)
    return set(matches)


def inject_missing_known_prefixes(text: str) -> tuple[str, list[str]]:
    """Inject known prefix declarations when terms use undeclared prefixes."""
    declared = extract_declared_prefixes(text)
    used = extract_used_prefixes(text)
    missing = sorted((used - declared) & set(KNOWN_PREFIXES))
    if not missing:
        return text, []

    injections = [f"@prefix {p}: <{KNOWN_PREFIXES[p]}> ." for p in missing]
    updated = "\n".join(injections) + "\n" + text.lstrip()
    return updated, missing


def validate_and_normalize_ontology(raw_text: str, rdf_format: str) -> tuple[str, list[str]]:
    """Normalize and validate returned ontology text."""
    normalized = raw_text.strip()
    added_prefixes: list[str] = []
    if rdf_format in {"turtle", "n3"}:
        normalized, added_prefixes = inject_missing_known_prefixes(normalized)
    parse_graph(normalized, rdf_format)
    return normalized, added_prefixes


def request_ontology_edit(prompt: str, model: str) -> str:
    """Call OpenAI and return ontology text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        openai_module = importlib.import_module("openai")
        openai_client = getattr(openai_module, "OpenAI")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'openai' package is required. Install it with: uv add openai"
        ) from exc

    client = openai_client(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You edit OWL ontologies. Return only ontology text in the requested "
                    "serialization format. Do not add markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("OpenAI returned an empty response.")
    return strip_markdown_fences(content)


def request_ontology_syntax_fix(
    ontology_text: str,
    rdf_format: str,
    parse_error: Exception,
    model: str,
) -> str:
    """Ask OpenAI for a syntax-only repair of ontology text."""
    prompt = dedent(
        f"""\
        Fix the syntax of this ontology serialization.

        Requirements:
        - Return only ontology text in {rdf_format}.
        - Preserve meaning; only fix syntax/prefix issues.
        - Ensure all used prefixes are declared.
        - Do not include markdown fences.

        Parser error:
        {parse_error}

        Ontology text:
        {ontology_text}
        """
    )
    return request_ontology_edit(prompt=prompt, model=model)


def init_state() -> None:
    """Initialize app session state keys."""
    st.session_state.setdefault("ontology_text", EXAMPLE_OWL)
    st.session_state.setdefault("rdf_format", "turtle")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_llm_prompt", "")
    st.session_state.setdefault("last_llm_response", "")
    st.session_state.setdefault("last_normalized_response", "")
    st.session_state.setdefault("openai_model", DEFAULT_OPENAI_MODEL)


def main() -> None:
    st.set_page_config(page_title="Ontology Chat Draft", layout="wide")
    st.title("Ontology Chat + Mermaid (Draft)")
    st.caption("Prototype UI for OWL editing with chat-driven change requests.")

    init_state()

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Ontology Text")
        st.session_state.rdf_format = st.selectbox(
            "RDF format",
            options=["turtle", "xml", "nt", "n3"],
            index=["turtle", "xml", "nt", "n3"].index(st.session_state.rdf_format),
        )
        st.session_state.openai_model = st.text_input(
            "OpenAI model",
            value=st.session_state.openai_model,
            help="Requires OPENAI_API_KEY in environment.",
        )
        st.session_state.ontology_text = st.text_area(
            "Edit ontology",
            value=st.session_state.ontology_text,
            height=340,
        )

        st.subheader("Chat")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_request = st.chat_input("Describe ontology change...")
        if user_request:
            st.session_state.messages.append({"role": "user", "content": user_request})
            prompt = draft_llm_prompt(
                user_request=user_request,
                ontology_text=st.session_state.ontology_text,
                rdf_format=st.session_state.rdf_format,
            )
            st.session_state.last_llm_prompt = prompt
            try:
                with st.spinner("Requesting ontology update from OpenAI..."):
                    model_name = st.session_state.openai_model.strip() or DEFAULT_OPENAI_MODEL
                    edited_ontology = request_ontology_edit(
                        prompt=prompt,
                        model=model_name,
                    )
                st.session_state.last_llm_response = edited_ontology
                try:
                    normalized_ontology, added_prefixes = validate_and_normalize_ontology(
                        edited_ontology, st.session_state.rdf_format
                    )
                except Exception as parse_exc:  # noqa: BLE001
                    with st.spinner("Attempting syntax repair..."):
                        repaired = request_ontology_syntax_fix(
                            ontology_text=edited_ontology,
                            rdf_format=st.session_state.rdf_format,
                            parse_error=parse_exc,
                            model=model_name,
                        )
                    st.session_state.last_llm_response = repaired
                    normalized_ontology, added_prefixes = validate_and_normalize_ontology(
                        repaired, st.session_state.rdf_format
                    )

                st.session_state.ontology_text = normalized_ontology
                st.session_state.last_normalized_response = normalized_ontology
                prefix_note = ""
                if added_prefixes:
                    prefix_note = f" Added missing prefixes: {', '.join(added_prefixes)}."
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Applied OpenAI ontology update and refreshed Mermaid diagram."
                            f"{prefix_note}"
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "OpenAI request failed after validation/repair attempts: "
                            f"{exc}"
                        ),
                    }
                )
            st.rerun()

        with st.expander("Last drafted LLM prompt", expanded=False):
            st.code(st.session_state.last_llm_prompt or "No prompt drafted yet.", language="text")
        with st.expander("Last OpenAI response", expanded=False):
            st.code(st.session_state.last_llm_response or "No model response yet.", language="text")
        with st.expander("Last normalized ontology", expanded=False):
            st.code(
                st.session_state.last_normalized_response or "No normalized ontology yet.",
                language="text",
            )

    with right_col:
        st.subheader("Mermaid Render")
        try:
            graph = parse_graph(st.session_state.ontology_text, st.session_state.rdf_format)
            schema = extract_schema(graph)
            mermaid = to_mermaid(schema)
            render_mermaid(mermaid)
            with st.expander("Mermaid source", expanded=False):
                st.code(mermaid, language="text")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse ontology: {exc}")
            st.info("Check RDF format and ontology syntax in the left panel.")


if __name__ == "__main__":
    main()
