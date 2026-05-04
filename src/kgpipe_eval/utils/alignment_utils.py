from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2ClassificationHead
from kgpipe.common import KG
from typing import TYPE_CHECKING, Literal, NamedTuple, Optional
from functools import lru_cache
from pydantic import BaseModel, ConfigDict, model_validator

from kgpipe_eval.utils.kg_utils import TripleGraph, Term, Triple, KgLike, KgManager
from kgpipe.util.embeddings.st_emb import get_model

from rdflib import RDFS, RDF
from rdflib.term import BNode
from rdflib.term import Literal as RdLiteral
from kgpipe.datasets.multipart_multisource import read_entities_csv, EntitiesRow
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tqdm import tqdm
# TODO source entities csv to label only graph

DEBUG = True

class EntityAlignmentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    method: Literal["label_embedding", "label_alias_embedding", "label_embedding_and_type"] = "label_embedding"
    reference_kg: Optional[KgLike] = None
    verified_entities_path: Optional[Path] = None
    verified_entities_delimiter: str = "\t"
    entity_sim_threshold: float = 0.95

    # value_sim_threshold: float = 0.5

    @model_validator(mode="after")
    def _require_reference_source(self):
        if self.reference_kg is None and self.verified_entities_path is None:
            raise ValueError("Provide either `reference_kg` or `verified_entities_path`.")
        return self


EntityAlignment = NamedTuple("EntityAlignment", [("source", Term), ("target", Term), ("score", float)])
TripleAlignment = NamedTuple("TripleAlignment", [("source", Triple), ("target", Triple)])

# Core alignment method interfaces

@lru_cache(maxsize=1000)
def get_aligned_entities(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[EntityAlignment]:
    return kg.entities.intersection(reference_kg.entities)

def get_aligned_triples(kg: KG, reference_kg: KG, method: Literal["exact", "fuzzy", "semantic"] = "exact") -> list[TripleAlignment]:
    return kg.triples.intersection(reference_kg.triples)

# Helper methods

# def get_entity_uri_label_pairs(triple_graph: TripleGraph) -> list[tuple[Term, Term]]:
#     return [(s, label) for s, _, label in triple_graph.triples((None, RDFS.label, None))]

UriLabelTypePair = NamedTuple("UriLabelTypePair", [("uri", Term), ("label", Term), ("type", Term)])

def get_entity_uri_label_type_pairs(kg: KG) -> list[UriLabelTypePair]:
    label_by_uri = {}
    type_by_uri = {}
    for s, p, o in kg.triples((None, RDFS.label, None)):
        label_by_uri[str(s)] = str(o)
    for s, p, o in kg.triples((None, RDF.type, None)):
        type_by_uri[str(s)] = str(o)
    for uri in label_by_uri:
        if uri in type_by_uri:
            yield UriLabelTypePair(uri=uri, label=label_by_uri[uri], type=type_by_uri[uri])
        else:
            yield UriLabelTypePair(uri=uri, label=label_by_uri[uri], type=None)

def load_verified_entities(path: Path, delimiter: str = "\t") -> list[UriLabelTypePair]:
    """
    """
    if path.name.endswith(".json"):
        raise ValueError("JSON format not supported for verified entities")
    elif path.name.endswith(".csv"):
        return [UriLabelTypePair(uri=entity.entity_id, label=entity.entity_label, type=entity.entity_type) for entity in read_entities_csv(path=path, delimiter=delimiter)]
    else:
        raise ValueError(f"Unsupported file type: {path}")

def load_entity_uri_label_type_pairs(config: EntityAlignmentConfig) -> list[UriLabelTypePair]:
    if config.verified_entities_path is not None:
        return load_verified_entities(config.verified_entities_path, delimiter=config.verified_entities_delimiter)
    elif config.reference_kg is not None:
        # `get_entity_uri_label_type_pairs` is a generator; downstream alignment uses indexing.
        return list(get_entity_uri_label_type_pairs(KgManager.load_kg(config.reference_kg)))
    else:
        raise ValueError("No verified entities path or reference KG provided")

# Specific alignment methods

def align_entities_by_label_embedding(tg: TripleGraph, config: EntityAlignmentConfig) -> list[EntityAlignment]:
    model = get_model()
    ref_entity_uri_label_type_pairs = load_entity_uri_label_type_pairs(config)
    ref_labels = [pair.label for pair in ref_entity_uri_label_type_pairs]
    ref_labels_embeddings = model.encode(ref_labels, convert_to_numpy=True, show_progress_bar=False)

    gen_entity_uri_label_type_pairs = list(get_entity_uri_label_type_pairs(tg))
    gen_labels = [pair.label for pair in gen_entity_uri_label_type_pairs]
    gen_labels_embeddings = model.encode(gen_labels, convert_to_numpy=True, show_progress_bar=False)


    sims = np.dot(gen_labels_embeddings, ref_labels_embeddings.T)

    alignments = []
    for i in range(sims.shape[0]):
        best_j = np.argmax(sims[i])
        if sims[i][best_j] >= config.entity_sim_threshold:
            alignments.append(EntityAlignment(source=gen_entity_uri_label_type_pairs[i].uri, target=ref_entity_uri_label_type_pairs[best_j].uri, score=sims[i][best_j]))
    return alignments

def align_by_label_alias_embedding(triple_graph: TripleGraph, model="", similarity="cosine", threshold=0.5):
    pass


if TYPE_CHECKING:  # avoid circular import at runtime
    from kgpipe_eval.metrics.triple_alignment import TripleAlignmentConfig


def _is_literal(term: Term) -> bool:
    return isinstance(term, RdLiteral)


def _literal_text(lit: RdLiteral) -> str:
    # Prefer lexical form; fall back to python value string.
    try:
        return str(lit)
    except Exception:
        return str(lit.toPython())


def align_triples_by_value_embedding(tg: TripleGraph, config: "TripleAlignmentConfig") -> list[TripleAlignment]:
    """
    Align generated triples in `tg` to reference triples using:
    - entity alignment (for URI/BNode subjects/objects)
    - embedding similarity for literal object values (for same subject+predicate)
    """
    ref_tg = KgManager.load_kg(config.reference_kg)

    # 0) Blank node mapping.
    #
    # rdflib assigns fresh IDs to BNodes on parse/load, so loading the "same" KG
    # twice will not preserve BNode identifiers. We map BNodes by an outgoing-edge
    # signature (predicate + object lexical form) to make exact-equal graphs align.
    def _term_key(t: Term) -> str:
        return str(t)

    def _bnode_signature(g: TripleGraph, b: BNode) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        for _, p, o in g.triples((b, None, None)):
            if _is_literal(o):
                ok = _literal_text(o)
            else:
                ok = _term_key(o)
            pairs.append((_term_key(p), ok))
        pairs.sort()
        return tuple(pairs)

    def _build_bnode_map(gen_g: TripleGraph, ref_g: TripleGraph) -> dict[str, Term]:
        ref_by_sig: dict[tuple[tuple[str, str], ...], list[BNode]] = {}
        for s, _, _ in ref_g.triples((None, None, None)):
            if isinstance(s, BNode):
                print(f"Ref bnode: {s}")
                sig = _bnode_signature(ref_g, s)
                ref_by_sig.setdefault(sig, []).append(s)

        gen_by_sig: dict[tuple[tuple[str, str], ...], list[BNode]] = {}
        for s, _, _ in gen_g.triples((None, None, None)):
            if isinstance(s, BNode):
                print(f"Gen bnode: {s}")
                sig = _bnode_signature(gen_g, s)
                gen_by_sig.setdefault(sig, []).append(s)

        # Accept signature matches. If a signature occurs multiple times in both graphs,
        # map deterministically by sorting node IDs and zipping. This makes identical KGs
        # align even when they contain repeated blank-node structures.
        out: dict[str, Term] = {}
        for sig, gen_nodes in gen_by_sig.items():
            ref_nodes = ref_by_sig.get(sig, [])
            if not ref_nodes:
                continue
            if len(gen_nodes) != len(ref_nodes):
                continue
            for gnode, rnode in zip(sorted(gen_nodes, key=_term_key), sorted(ref_nodes, key=_term_key)):
                out[_term_key(gnode)] = rnode
        return out

    gen_bnode_to_ref: dict[str, Term] = _build_bnode_map(tg, ref_tg)
    
    # 1) Entity alignments (generated -> reference)
    ent_cfg = config.entity_alignment_config
    if getattr(ent_cfg, "reference_kg", None) is None and getattr(ent_cfg, "verified_entities_path", None) is None:
        # Ensure validator requirements are met; default to using the reference KG.
        ent_cfg = ent_cfg.model_copy(update={"reference_kg": config.reference_kg})

    entity_alignments = align_entities_by_label_embedding(tg, ent_cfg)

    if DEBUG: print("Entity alignments: ", len(entity_alignments))

    gen_to_ref_entity: dict[str, Term] = {}
    best_score_by_gen: dict[str, float] = {}
    for a in entity_alignments:
        gen_key = str(a.source)
        if gen_key not in best_score_by_gen or a.score > best_score_by_gen[gen_key]:
            best_score_by_gen[gen_key] = float(a.score)
            gen_to_ref_entity[gen_key] = a.target

    # 2) Index generated triples, both raw and entity-mapped
    mapped_gen_triples: list[tuple[Triple, Triple]] = []  # (raw_gen, mapped_to_ref_space)
    gen_by_sp_literal: dict[tuple[Term, Term], list[tuple[Triple, str]]] = {}
    gen_by_sp_entity: dict[tuple[Term, Term], set[Triple]] = {}

    if DEBUG: print("Gen by sp literal: ", len(gen_by_sp_literal))
    if DEBUG: print("Gen by sp entity: ", len(gen_by_sp_entity))

    sp_iter = getattr(tg, "iter_sp_groups", None)
    if callable(sp_iter):
        sp_groups = sp_iter()
        for s, p, os in sp_groups:
            for o in os:
                mapped_s = gen_to_ref_entity.get(str(s), gen_bnode_to_ref.get(str(s), s))
                mapped_o = gen_to_ref_entity.get(str(o), gen_bnode_to_ref.get(str(o), o)) if not _is_literal(o) else o
                mapped = (mapped_s, p, mapped_o)
                raw = (s, p, o)
                mapped_gen_triples.append((raw, mapped))

                # Normalize keys to string form to avoid rdflib Term vs string mismatches.
                sp = (_term_key(mapped_s), _term_key(p))
                if _is_literal(o):
                    gen_by_sp_literal.setdefault(sp, []).append((raw, _literal_text(o)))
                else:
                    gen_by_sp_entity.setdefault(sp, set()).add(raw)
    else:
        for s, p, o in tg.triples((None, None, None)):
            mapped_s = gen_to_ref_entity.get(str(s), gen_bnode_to_ref.get(str(s), s))
            mapped_o = gen_to_ref_entity.get(str(o), gen_bnode_to_ref.get(str(o), o)) if not _is_literal(o) else o
            mapped = (mapped_s, p, mapped_o)
            raw = (s, p, o)
            mapped_gen_triples.append((raw, mapped))

            # Normalize keys to string form to avoid rdflib Term vs string mismatches.
            sp = (_term_key(mapped_s), _term_key(p))
            if _is_literal(o):
                gen_by_sp_literal.setdefault(sp, []).append((raw, _literal_text(o)))
            else:
                gen_by_sp_entity.setdefault(sp, set()).add(raw)

    if DEBUG: print("Mapped gen triples: ", len(mapped_gen_triples))

    # 3) Prepare literal embedding caches (optional)
    model = get_model()
    alignments: list[TripleAlignment] = []

    # Encode generated literal texts once, cache by text.
    cache_gen_literals = bool(getattr(config, "cache_literal_embeddings", True))
    gen_lit_emb_by_text: dict[str, np.ndarray] = {}
    if cache_gen_literals and gen_by_sp_literal:
        unique_texts = sorted({txt for candidates in gen_by_sp_literal.values() for _, txt in candidates})
        if unique_texts:
            emb = model.encode(unique_texts, convert_to_numpy=True, show_progress_bar=True)
            gen_lit_emb_by_text = {t: emb[i : i + 1] for i, t in enumerate(unique_texts)}

    # Reference literal embedding cache (by text).
    cache_ref_literals = bool(getattr(config, "cache_ref_literal_embeddings", True))
    ref_lit_emb_by_text: dict[str, np.ndarray] = {}
    if cache_ref_literals:
        unique_texts = sorted({_literal_text(ro) for _, _, ro in ref_tg.triples((None, None, None))})
        if unique_texts:
            emb = model.encode(unique_texts, convert_to_numpy=True, show_progress_bar=True)
            ref_lit_emb_by_text = {t: emb[i : i + 1] for i, t in enumerate(unique_texts)}

    def get_ref_literal_embedding(texts: list[str]) -> np.ndarray:
        if cache_ref_literals and ref_lit_emb_by_text:
            return np.concatenate([ref_lit_emb_by_text[t] for t in texts], axis=0)
        else:
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


    def get_gen_literal_embedding(texts: list[str]) -> np.ndarray:
        if cache_gen_literals and gen_lit_emb_by_text:
            return np.concatenate([gen_lit_emb_by_text[t] for t in texts], axis=0)
        else:
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    if DEBUG: print("gen_lit_emb_by_text: ", len(gen_lit_emb_by_text))
    if DEBUG: print("ref_lit_emb_by_text: ", len(ref_lit_emb_by_text))

    from rdflib import Graph, URIRef
    gen_graph : Graph = tg._graph()
    ref_graph : Graph = ref_tg._graph()

    test_objects = list(ref_graph.objects(URIRef("http://kg.org/resource/f4eb17c4ed78c87c29124018c9f180b5"), URIRef("http://kg.org/ontology/deathPlace"), unique=True))
    if DEBUG: print("Test objects: ", len(test_objects))

    if DEBUG: print("Gen graph: ", len(list(gen_graph.triples((None, None, None)))))
    if DEBUG: print("Ref graph: ", len(list(ref_graph.triples((None, None, None)))))

    for gs, gp in tqdm(gen_graph.subject_predicates(unique=True), desc="Aligning triples by value embedding"):
        # sp = (_term_key(gs), _term_key(gp))

        # check for s mapping in reference space
        ref_s = gen_to_ref_entity.get(str(gs), gen_bnode_to_ref.get(str(gs), gs))
        if ref_s is None:
            continue # s is not mapped to reference space

        gen_objects = list(gen_graph.objects(gs, gp))
        gen_literal_objs = [o for o in gen_objects if _is_literal(o)]

        # IMPORTANT: query reference objects in reference-space subject
        ref_objects = list(ref_graph.objects(URIRef(str(ref_s)), URIRef(str(gp))))
        ref_literal_objs = [o for o in ref_objects if _is_literal(o)]

        # print("gs: ", gs, "gp: ", gp)
        # print("ref_s: ", ref_s)
        # print("gen_literal_objs: ", len(gen_literal_objs))
        # print("ref_literal_objs: ", len(ref_literal_objs))
        # print("gen_objects: ", len(gen_objects))
        # print("ref_objects: ", len(ref_objects))

        if len(gen_literal_objs) > 0 and len(ref_literal_objs) > 0:

            gen_object_texts = [_literal_text(o) for o in gen_literal_objs]
            ref_object_texts = [_literal_text(o) for o in ref_literal_objs]

            gen_object_embeddings = get_gen_literal_embedding(gen_object_texts)
            ref_object_embeddings = get_ref_literal_embedding(ref_object_texts)

            sims = np.dot(gen_object_embeddings, ref_object_embeddings.T)  # shape (n_gen, n_ref)
            best_flat = int(np.argmax(sims))
            best_i, best_j = np.unravel_index(best_flat, sims.shape)

            if float(sims[best_i, best_j]) >= float(config.value_sim_threshold):
                alignments.append(
                    TripleAlignment(
                        source=(gs, gp, gen_literal_objs[best_i]),
                        target=(ref_s, gp, ref_literal_objs[best_j]),
                    )
                )

        # get all non-literal objects mapped to reference space
        gen_object_non_literal = [o for o in gen_objects if not _is_literal(o)]
        ref_object_non_literal = [o for o in ref_objects if not _is_literal(o)]

        # find if any of the non-literal objects in the generated graph are mapped to the same object in the reference graph
        for gen_obj in gen_object_non_literal:
            if gen_to_ref_entity.get(str(gen_obj), gen_bnode_to_ref.get(str(gen_obj), gen_obj)) in ref_object_non_literal:
                alignments.append(TripleAlignment(source=(gs, gp, gen_obj), target=(gs, gp, ref_object_non_literal[ref_object_non_literal.index(gen_obj)])))

    return alignments