"""
Custom relation linker: relation phrases → clusters → (optionally) KG predicates.

Pipeline:
  1. SBERT embeddings of relation phrases
  2. Cluster phrases (e.g. HDBSCAN / Agglomerative)
  3. (Optional) LLM or rule-based mapping of clusters → KG predicates

Using description and alt labels for predicates is recommended: they give more
surface forms for similarity (SBERT) and for rule/LLM mapping (e.g. "directed by"
vs skos:altLabel / rdfs:comment).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class PredicateInfo:
    """One KG predicate with optional text for matching."""

    iri: str
    label: str
    description: str | None = None
    alt_labels: list[str] = field(default_factory=list)

    def surface_forms(self) -> list[str]:
        """All text forms usable for matching (label + description + alt_labels)."""
        forms = [self.label]
        if self.description:
            forms.append(self.description)
        forms.extend(self.alt_labels)
        return [f for f in forms if f and f.strip()]


@dataclass
class RelationPhrase:
    """A relation phrase extracted from text (e.g. "directed by", "starred in")."""

    text: str
    span: tuple[int, int] | None = None  # optional char span in source
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterAssignment:
    """Result of clustering: phrase index → cluster id."""

    phrase_index: int
    cluster_id: int
    phrase: RelationPhrase


@dataclass
class PredicateMapping:
    """Mapping from a cluster (or phrase) to a KG predicate."""

    cluster_id: int
    predicate_iri: str
    predicate_label: str
    score: float = 1.0
    method: str = "rule"  # "rule" | "llm"


# ---------------------------------------------------------------------------
# Step 1: SBERT embeddings
# ---------------------------------------------------------------------------


def embed_relation_phrases(
    phrases: list[RelationPhrase],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """
    Embed relation phrases with SBERT (sentence-transformers).

    Returns array of shape (n_phrases, embedding_dim).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for SBERT embeddings. "
            "Install with: pip install sentence-transformers"
        ) from None

    model = SentenceTransformer(model_name, device=device)
    texts = [p.text.strip() for p in phrases]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 2: Clustering
# ---------------------------------------------------------------------------


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "agglomerative",
    n_clusters: int | None = None,
    min_cluster_size: int = 2,
    min_samples: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Cluster embeddings. Returns cluster labels, shape (n_phrases,).

    - method "agglomerative": needs n_clusters or infers from distance_threshold.
    - method "hdbscan": density-based; use min_cluster_size (and optionally min_samples).
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=np.int64)

    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        n_clusters = n_clusters or max(2, n // 5)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
            **kwargs,
        )
        clustering.fit(embeddings)
        return clustering.labels_.astype(np.int64)

    if method == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for method='hdbscan'. "
                "Install with: pip install hdbscan"
            ) from None
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            **kwargs,
        )
        clusterer.fit(embeddings)
        # HDBSCAN uses -1 for noise; we keep that
        return clusterer.labels_.astype(np.int64)

    raise ValueError(f"Unknown clustering method: {method}")


# ---------------------------------------------------------------------------
# Step 3a: Rule-based mapping (cluster → predicate)
# ---------------------------------------------------------------------------


def _embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required for rule-based mapping") from None
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True)


def map_clusters_to_predicates_rule(
    cluster_centroids: np.ndarray,
    cluster_ids: list[int],
    phrases_per_cluster: dict[int, list[RelationPhrase]],
    predicates: list[PredicateInfo],
    model_name: str = "all-MiniLM-L6-v2",
    use_description_and_alt_labels: bool = True,
) -> list[PredicateMapping]:
    """
    Map each cluster to the best-matching predicate by cosine similarity of
    cluster centroid vs predicate surface forms.

    cluster_centroids: shape (len(cluster_ids), dim), same order as cluster_ids.
    If use_description_and_alt_labels is True, each predicate is represented by
    the mean of embeddings of label + description + alt_labels (recommended).
    """
    from numpy.linalg import norm

    # Embed predicate surface forms (one embedding per (predicate, form) for best match)
    predicate_texts: list[str] = []
    predicate_iris_labels: list[tuple[str, str]] = []
    for pred in predicates:
        forms = pred.surface_forms() if use_description_and_alt_labels else [pred.label]
        for f in forms:
            if f and f.strip():
                predicate_texts.append(f.strip())
                predicate_iris_labels.append((pred.iri, pred.label))

    if not predicate_texts:
        return []

    pred_embeddings = _embed_texts(predicate_texts, model_name=model_name)
    out: list[PredicateMapping] = []

    for idx, cid in enumerate(cluster_ids):
        cen = cluster_centroids[idx] if cluster_centroids[idx].ndim == 1 else cluster_centroids[idx].mean(axis=0)
        cen_norm = norm(cen)
        if cen_norm <= 0:
            continue

        best_iri: str | None = None
        best_label = ""
        best_score = -1.0

        for j in range(len(predicate_texts)):
            pnorm = norm(pred_embeddings[j]) or 1e-9
            sim = float(np.dot(cen, pred_embeddings[j]) / (cen_norm * pnorm))
            if sim > best_score:
                best_score = sim
                best_iri, best_label = predicate_iris_labels[j]

        if best_iri is not None and best_score > 0:
            out.append(
                PredicateMapping(
                    cluster_id=cid,
                    predicate_iri=best_iri,
                    predicate_label=best_label,
                    score=best_score,
                    method="rule",
                )
            )

    return out


# ---------------------------------------------------------------------------
# Step 3b: LLM mapping (optional)
# ---------------------------------------------------------------------------


def map_clusters_to_predicates_llm(
    phrases_per_cluster: dict[int, list[RelationPhrase]],
    predicates: list[PredicateInfo],
    *,
    prompt_builder: Callable[
        [dict[int, list[RelationPhrase]], list[PredicateInfo]], str
    ] | None = None,
    invoke_llm: Callable[[str], str] | None = None,
    parse_response: Callable[[str], list[tuple[int, str]]] | None = None,
) -> list[PredicateMapping]:
    """
    Map clusters to predicates via an LLM.

    You must supply either (prompt_builder, invoke_llm, parse_response) or
    implement this externally and call the linker with pre-filled mappings.

    Default placeholder: returns no mappings; override with your LLM client.
    """
    if prompt_builder is None or invoke_llm is None or parse_response is None:
        return []

    prompt = prompt_builder(phrases_per_cluster, predicates)
    response = invoke_llm(prompt)
    pairs = parse_response(response)  # list of (cluster_id, predicate_iri_or_label)

    result: list[PredicateMapping] = []
    for cid, iri_or_label in pairs:
        pred = next(
            (p for p in predicates if p.iri == iri_or_label or p.label == iri_or_label),
            None,
        )
        if pred is not None:
            result.append(
                PredicateMapping(
                    cluster_id=cid,
                    predicate_iri=pred.iri,
                    predicate_label=pred.label,
                    score=1.0,
                    method="llm",
                )
            )
    return result


# ---------------------------------------------------------------------------
# End-to-end linker
# ---------------------------------------------------------------------------


@dataclass
class RelationLinkerResult:
    """Full result of the relation linker pipeline."""

    phrases: list[RelationPhrase]
    embeddings: np.ndarray
    cluster_labels: np.ndarray
    phrases_per_cluster: dict[int, list[RelationPhrase]]
    cluster_assignments: list[ClusterAssignment]
    predicate_mappings: list[PredicateMapping]

    def phrase_to_predicate(
        self,
        phrase_index: int,
        default_iri: str | None = None,
    ) -> tuple[str | None, str | None, float]:
        """
        Resolve a phrase index to (predicate_iri, predicate_label, score).
        Uses cluster → predicate mapping; returns default_iri if unmapped.
        """
        label = self.cluster_labels[phrase_index]
        if label < 0:
            return (default_iri, None, 0.0)
        mapping = next((m for m in self.predicate_mappings if m.cluster_id == label), None)
        if mapping is None:
            return (default_iri, None, 0.0)
        return (mapping.predicate_iri, mapping.predicate_label, mapping.score)


def run_relation_linker(
    phrases: list[RelationPhrase],
    predicates: list[PredicateInfo],
    *,
    embed_model: str = "all-MiniLM-L6-v2",
    cluster_method: str = "agglomerative",
    n_clusters: int | None = None,
    min_cluster_size: int = 2,
    map_to_predicates: bool = True,
    use_rule_mapping: bool = True,
    use_description_and_alt_labels: bool = True,
    cluster_embeddings_for_mapping: np.ndarray | None = None,
) -> RelationLinkerResult:
    """
    Run the full pipeline: embed → cluster → (optionally) map to predicates.

    - If map_to_predicates is True and use_rule_mapping is True, clusters are
      mapped to the given predicates via rule-based similarity (recommended to
      set use_description_and_alt_labels True for better matching).
    - cluster_embeddings_for_mapping: if provided, must be one vector per
      cluster id (used for rule mapping); otherwise we compute centroid per
      cluster from phrase embeddings.
    """
    if not phrases:
        return RelationLinkerResult(
            phrases=[],
            embeddings=np.zeros((0, 0), dtype=np.float32),
            cluster_labels=np.array([], dtype=np.int64),
            phrases_per_cluster={},
            cluster_assignments=[],
            predicate_mappings=[],
        )

    # Step 1: SBERT embeddings
    embeddings = embed_relation_phrases(phrases, model_name=embed_model)

    # Step 2: Cluster
    cluster_labels = cluster_embeddings(
        embeddings,
        method=cluster_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
    )

    # Build phrases_per_cluster and assignments
    phrases_per_cluster: dict[int, list[RelationPhrase]] = {}
    cluster_assignments: list[ClusterAssignment] = []
    for i, (phrase, cid) in enumerate(zip(phrases, cluster_labels)):
        cid_int = int(cid)
        cluster_assignments.append(ClusterAssignment(phrase_index=i, cluster_id=cid_int, phrase=phrase))
        phrases_per_cluster.setdefault(cid_int, []).append(phrase)

    # Centroid per cluster (for rule-based mapping)
    if cluster_embeddings_for_mapping is None:
        unique_ids = sorted(set(cluster_labels) - {-1})
        cluster_embeddings_for_mapping = np.zeros(
            (max(unique_ids) + 1 if unique_ids else 0, embeddings.shape[1]),
            dtype=np.float32,
        )
        for cid in unique_ids:
            mask = cluster_labels == cid
            cluster_embeddings_for_mapping[cid] = embeddings[mask].mean(axis=0)

    # Step 3: Map clusters → predicates
    predicate_mappings: list[PredicateMapping] = []
    if map_to_predicates and use_rule_mapping and predicates:
        # Build cluster centroid array indexed by cluster id
        unique_ids = sorted(k for k in phrases_per_cluster if k >= 0)
        if unique_ids:
            cen_list = [cluster_embeddings_for_mapping[cid] for cid in unique_ids]
            cen_arr = np.array(cen_list)
            predicate_mappings = map_clusters_to_predicates_rule(
                cen_arr,
                unique_ids,
                phrases_per_cluster,
                predicates,
                model_name=embed_model,
                use_description_and_alt_labels=use_description_and_alt_labels,
            )

    return RelationLinkerResult(
        phrases=phrases,
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        phrases_per_cluster=phrases_per_cluster,
        cluster_assignments=cluster_assignments,
        predicate_mappings=predicate_mappings,
    )


# ---------------------------------------------------------------------------
# Convenience: build PredicateInfo from typical KG metadata
# ---------------------------------------------------------------------------


def predicate_info(
    iri: str,
    label: str,
    description: str | None = None,
    alt_labels: list[str] | None = None,
) -> PredicateInfo:
    """Build PredicateInfo; alt_labels can be a list or comma-separated string."""
    if alt_labels is None:
        al: list[str] = []
    elif isinstance(alt_labels, str):
        al = [x.strip() for x in alt_labels.split(",") if x.strip()]
    else:
        al = list(alt_labels)
    return PredicateInfo(iri=iri, label=label, description=description, alt_labels=al)
