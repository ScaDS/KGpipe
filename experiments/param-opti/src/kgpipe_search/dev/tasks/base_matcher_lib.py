from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from rdflib import Graph, Literal, RDFS, URIRef
from sentence_transformers import util

from kgpipe.common import Data
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document, ER_Match

# Reuse the shared embedder/model cache from the linker implementation.
from param_opti.tasks.base_linker_lib import SentenceTransformerEmbedder, _validate_embedding_dimensions


def _normalize_label(text: str) -> str:
    return " ".join(text.replace("_", " ").replace("-", " ").strip().lower().split())


def _safe_first_literal(values: Iterable[object]) -> Optional[str]:
    for v in values:
        if isinstance(v, Literal):
            s = str(v).strip()
            if s:
                return s
    return None


def _fallback_label_from_uri(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class _LabeledUri:
    uri: URIRef
    label: str


def _extract_labeled_entities(graph: Graph) -> List[_LabeledUri]:
    """
    Extract subject/object URIRefs that have an rdfs:label.
    """
    uris: set[URIRef] = set()
    for s, _, o in graph:
        if isinstance(s, URIRef):
            uris.add(s)
        if isinstance(o, URIRef):
            uris.add(o)

    labeled: List[_LabeledUri] = []
    for u in uris:
        label = _safe_first_literal(graph.objects(u, RDFS.label))
        if label:
            labeled.append(_LabeledUri(u, label))
    return labeled


def _extract_labeled_predicates(graph: Graph) -> List[_LabeledUri]:
    """
    Extract predicate URIRefs and use rdfs:label if present, otherwise fall back to local-name.
    """
    preds: set[URIRef] = {p for _, p, _ in graph if isinstance(p, URIRef)}
    labeled: List[_LabeledUri] = []
    for p in preds:
        label = _safe_first_literal(graph.objects(p, RDFS.label)) or _fallback_label_from_uri(p)
        labeled.append(_LabeledUri(p, label))
    return labeled


def _best_matches(
    source: Sequence[_LabeledUri],
    target: Sequence[_LabeledUri],
    *,
    model_name: str,
    threshold: float,
    id_type: str,
) -> List[ER_Match]:
    if not source or not target:
        return []

    embedder = SentenceTransformerEmbedder(model_name=model_name)
    src_texts = [_normalize_label(x.label) for x in source]
    tgt_texts = [_normalize_label(x.label) for x in target]

    src_emb = embedder.encode(src_texts)
    tgt_emb = embedder.encode(tgt_texts)
    _validate_embedding_dimensions(src_emb, tgt_emb)

    sims = util.cos_sim(src_emb, tgt_emb)
    matches: List[ER_Match] = []

    for i, src in enumerate(source):
        best_idx = int(sims[i].argmax())
        best_score = float(sims[i][best_idx])
        if best_score < float(threshold):
            continue
        tgt = target[best_idx]
        matches.append(
            ER_Match(
                id_1=str(src.uri),
                id_2=str(tgt.uri),
                score=best_score,
                id_type=id_type,
            )
        )
    return matches


def _write_er_document(output_path: Path, matches: List[ER_Match]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ER_Document(matches=matches)
    output_path.write_text(doc.model_dump_json(), encoding="utf-8")


def _label_embedding_match_two_graphs(
    inputs: Dict[str, Data],
    outputs: Dict[str, Data],
    *,
    model_name: str,
    threshold: float,
    include_entities: bool,
    include_relations: bool,
) -> None:
    source_graph = Graph()
    source_graph.parse(inputs["source"].path, format="nt")

    target_graph = Graph()
    target_graph.parse(inputs["target"].path, format="nt")

    matches: List[ER_Match] = []

    if include_entities:
        source_entities = _extract_labeled_entities(source_graph)
        target_entities = _extract_labeled_entities(target_graph)
        matches.extend(
            _best_matches(
                source_entities,
                target_entities,
                model_name=model_name,
                threshold=threshold,
                id_type="entity",
            )
        )

    if include_relations:
        source_preds = _extract_labeled_predicates(source_graph)
        target_preds = _extract_labeled_predicates(target_graph)
        matches.extend(
            _best_matches(
                source_preds,
                target_preds,
                model_name=model_name,
                threshold=threshold,
                id_type="relation",
            )
        )

    _write_er_document(outputs["output"].path, matches)


def label_embedding_graph_alignment_match(
    inputs: Dict[str, Data],
    outputs: Dict[str, Data],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,
) -> None:
    """
    Align two RDF graphs: match subject/object entities by rdfs:label and predicates by label.

    Writes `ER_Document` JSON with both entity and relation matches (same shape as `paris_lib`).
    """
    _label_embedding_match_two_graphs(
        inputs,
        outputs,
        model_name=model_name,
        threshold=threshold,
        include_entities=True,
        include_relations=True,
    )


def label_embedding_entity_alignment_match(
    inputs: Dict[str, Data],
    outputs: Dict[str, Data],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,
) -> None:
    """Entity alignment only: matches with id_type \"entity\"."""
    _label_embedding_match_two_graphs(
        inputs,
        outputs,
        model_name=model_name,
        threshold=threshold,
        include_entities=True,
        include_relations=False,
    )


def label_embedding_relation_alignment_match(
    inputs: Dict[str, Data],
    outputs: Dict[str, Data],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,
) -> None:
    """Relation / predicate alignment only: matches with id_type \"relation\"."""
    _label_embedding_match_two_graphs(
        inputs,
        outputs,
        model_name=model_name,
        threshold=threshold,
        include_entities=False,
        include_relations=True,
    )


def label_embedding_graph_match(
    inputs: Dict[str, Data],
    outputs: Dict[str, Data],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,
) -> None:
    """Backward-compatible alias for full graph alignment (entities + relations)."""
    label_embedding_graph_alignment_match(
        inputs, outputs, model_name=model_name, threshold=threshold
    )

