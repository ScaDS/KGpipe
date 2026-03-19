from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import math
import re

from rdflib import Graph, RDF, RDFS, URIRef, Literal, BNode
from kgpipe_llm.common.core import get_client_from_env
from kgpipe.util.embeddings import global_encode
from pydantic import BaseModel
import dotenv

dotenv.load_dotenv()


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class TripleItem:
    s: Any
    p: Any
    o: Any
    triple_id: str

@dataclass
class HypothesisItem:
    triple: TripleItem
    hypothesis: str

@dataclass
class SentenceItem:
    sid: str
    text: str

# @dataclass
class Judgement(BaseModel):
    triple_id: str
    hypothesis: str
    entailed: bool
    evidence_sentence_ids: List[str]
    confidence: float
    reason: str

class JudgementResponse(BaseModel):
    judgement: List[Judgement]

# ----------------------------
# 1) RDF -> hypotheses (NL)
# ----------------------------

def _best_label(g: Graph, node: Any) -> str:
    if isinstance(node, Literal):
        return str(node)

    # rdfs:label
    lbls = list(g.objects(node, RDFS.label))
    if lbls:
        # prefer no-language label
        for l in lbls:
            if isinstance(l, Literal) and l.language is None:
                return str(l)
        return str(lbls[0])

    if isinstance(node, URIRef):
        # QName/localname fallback
        try:
            qn = g.namespace_manager.qname(node)
            return qn.split(":", 1)[1] if ":" in qn else qn
        except Exception:
            s = str(node)
            if "#" in s:
                return s.rsplit("#", 1)[1]
            if "/" in s:
                return s.rsplit("/", 1)[1]
            return s

    if isinstance(node, BNode):
        return f"_:{node}"

    return str(node)


def _predicate_to_phrase(g: Graph, p: Any) -> str:
    if p == RDF.type:
        return "is a"
    # Prefer label of predicate if present; else local name
    return _best_label(g, p)


def triple_to_hypothesis(g: Graph, t: TripleItem) -> str:
    s = _best_label(g, t.s)
    p = t.p
    o = _best_label(g, t.o)

    if p == RDF.type:
        return f"{s} is a {o}."
    pred = _predicate_to_phrase(g, p)

    # Simple heuristics to make nicer English for some datatypes
    if isinstance(t.o, Literal):
        # Possessive form tends to read better for attributes
        # e.g. "Alice\tage\t30" -> "Alice's age is 30."
        return f"{s}'s {pred} is {o}."
    else:
        return f"{s} {pred} {o}."


def graph_to_hypotheses(g: Graph, ignore_type_triples: bool = False) -> List[HypothesisItem]:
    triples: List[TripleItem] = []
    i = 0
    for s, p, o in g:
        # Skip labels; they’re for rendering
        if p == RDFS.label:
            continue
        if ignore_type_triples and p == RDF.type:
            continue
        triples.append(TripleItem(s=s, p=p, o=o, triple_id=f"t{i}"))
        i += 1

    # Stable order
    triples.sort(key=lambda x: (str(x.s), str(x.p), str(x.o)))

    return [HypothesisItem(triple=t, hypothesis=triple_to_hypothesis(g, t)) for t in triples]


# ----------------------------
# 2) Text -> sentences
# ----------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[SentenceItem]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    sents = []
    for idx, s in enumerate([p.strip() for p in parts if p.strip()]):
        sents.append(SentenceItem(sid=f"S{idx+1}", text=s))
    return sents


# ----------------------------
# 3) Embeddings retrieval
# ----------------------------
# Plug in your embedding provider here. Keep this interface:
#   embed_texts(list[str]) -> list[list[float]]
# If you use OpenAI / other provider, implement it in that function.

def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """
    TODO: Implement with your embeddings model/provider.
    Return one vector per input text (same order).
    """
    return global_encode(texts)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def retrieve_evidence(
    hypothesis: str,
    sentences: List[SentenceItem],
    sent_vecs: List[List[float]],
    hyp_vec: List[float],
    k: int = 4,
    add_neighbors: bool = True,
) -> List[SentenceItem]:
    scored = []
    for sitem, vec in zip(sentences, sent_vecs):
        scored.append((_cosine(hyp_vec, vec), sitem))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[: min(k, len(scored))]]

    if not add_neighbors:
        return top

    # Add +/- 1 neighbor context around each selected sentence
    sid_to_idx = {s.sid: i for i, s in enumerate(sentences)}
    idxs = set()
    for s in top:
        i = sid_to_idx[s.sid]
        idxs.add(i)
        if i - 1 >= 0:
            idxs.add(i - 1)
        if i + 1 < len(sentences):
            idxs.add(i + 1)

    return [sentences[i] for i in sorted(idxs)]


# ----------------------------
# 4) LLM NLI check
# ----------------------------
# Plug in your LLM call here. Keep this interface:
#   llm_json(prompt_dict) -> dict
# where prompt_dict contains fields you need; return parsed JSON.

def llm_json(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a structured prompt to the LLM and return parsed JSON output.
    Must return a Python dict parsed from the model's JSON output.
    """
    llm = get_client_from_env()
    
    # Extract components from the prompt dictionary
    task = prompt.get("task", "")
    instructions = prompt.get("instructions", "")
    hypothesis = prompt.get("hypothesis", "")
    
    # Handle both "evidence" and "cited_evidence" fields
    evidence = prompt.get("evidence") or prompt.get("cited_evidence", [])
    
    # Format the evidence sentences
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join([
            f"{item.get('id', '')}: {item.get('text', '')}"
            for item in evidence
        ])
    
    # Build the main prompt string
    prompt_str = f"""Task: {task}

Hypothesis:
{hypothesis}

Evidence:
{evidence_text}"""
    
    # Use instructions as system prompt for better context
    system_prompt = instructions if instructions else ""
    
    # Send to LLM with "json" format for flexible JSON response
    return llm.send_prompt(prompt_str, schema_class=Judgement, system_prompt=system_prompt)


def build_nli_prompt(hypothesis: str, evidence: List[SentenceItem]) -> Dict[str, Any]:
    """
    We keep this as a structured object; your llm_json() can turn it into messages.
    """
    return {
        "task": "nli_entailment_check",
        "instructions": (
            "Determine whether the Hypothesis is strictly ENTAILED by the Evidence.\n"
            "Rules:\n"
            "- Use ONLY the provided Evidence.\n"
            "- If the Evidence does not explicitly support the Hypothesis, return entailed=false.\n"
            "- If entailed=true, you MUST cite at least one evidence_sentence_id.\n"
            "- Do not guess or use outside knowledge.\n"
            "Return ONLY valid JSON with keys: entailed (bool), confidence (0..1), "
            "evidence_sentence_ids (list[str]), reason (string)."
        ),
        "hypothesis": hypothesis,
        "evidence": [{"id": s.sid, "text": s.text} for s in evidence],
    }


def judge_one(h: HypothesisItem, evidence: List[SentenceItem]) -> Judgement:
    prompt = build_nli_prompt(h.hypothesis, evidence)
    out = llm_json(prompt)

    entailed = bool(out.get("entailed", False))
    conf = float(out.get("confidence", 0.0))
    ev_ids = out.get("evidence_sentence_ids") or []
    reason = str(out.get("reason", ""))

    # Enforce evidence requirement
    if entailed and not ev_ids:
        entailed = False
        conf = min(conf, 0.49)
        reason = (reason + " [Invalid: no evidence ids cited]").strip()

    return Judgement(
        triple_id=h.triple.triple_id,
        hypothesis=h.hypothesis,
        entailed=entailed,
        evidence_sentence_ids=list(ev_ids),
        confidence=conf,
        reason=reason,
    )


def build_verifier_prompt(hypothesis: str, evidence: List[SentenceItem], cited_ids: List[str]) -> Dict[str, Any]:
    cited = [s for s in evidence if s.sid in set(cited_ids)]
    return {
        "task": "strict_verifier",
        "instructions": (
            "Verify entailment strictly.\n"
            "- Use ONLY the cited evidence sentences.\n"
            "- If there is any missing detail or ambiguity, return entailed=false.\n"
            "Return ONLY valid JSON with keys: entailed (bool), confidence (0..1), reason (string)."
        ),
        "hypothesis": hypothesis,
        "cited_evidence": [{"id": s.sid, "text": s.text} for s in cited],
    }


def verify(j: Judgement, evidence: List[SentenceItem]) -> Judgement:
    if not j.entailed:
        return j
    vp = build_verifier_prompt(j.hypothesis, evidence, j.evidence_sentence_ids)
    out = llm_json(vp)
    ok = bool(out.get("entailed", False))
    conf = float(out.get("confidence", j.confidence))
    reason = str(out.get("reason", j.reason))

    if not ok:
        j.entailed = False
        j.confidence = min(conf, 0.49)
        j.reason = reason
    else:
        j.confidence = conf
        j.reason = reason
    return j


# ----------------------------
# 5) End-to-end coverage
# ----------------------------

def check_coverage(
    g: Graph,
    text: str,
    ignore_type_triples: bool = False,
    k: int = 4,
    add_neighbors: bool = True,
    use_verifier: bool = True,
    split_sentences: bool = True,
) -> Dict[str, Any]:
    """
    Check the coverage of the text against the RDF graph.
    Args:
        g: The RDF graph.
        text: The text to check the coverage of.
        ignore_type_triples: Whether to ignore type triples.
        k: The number of evidence sentences to retrieve.
        add_neighbors: Whether to add neighboring sentences as evidence.
        use_verifier: Whether to use the verifier.
        split_sentences: Whether to split the text into sentences.

    Returns:
        A dictionary containing the coverage information.
    """
    hypotheses = graph_to_hypotheses(g, ignore_type_triples=ignore_type_triples)
    print(f"INFO: hypotheses {len(hypotheses)}")
    print([h.hypothesis for h in hypotheses])
    sentences = split_sentences(text) if split_sentences else [SentenceItem(sid="S1", text=text)]

    # Edge case: no sentences
    if not sentences:
        return {
            "covered_all": False,
            "coverage_rate": 0.0,
            "results": [
                {
                    "triple_id": h.triple.triple_id,
                    "hypothesis": h.hypothesis,
                    "entailed": False,
                    "evidence_sentence_ids": [],
                    "confidence": 0.0,
                    "reason": "Empty text.",
                }
                for h in hypotheses
            ],
        }

    # Embed once
    sent_texts = [s.text for s in sentences]
    hyp_texts = [h.hypothesis for h in hypotheses]

    sent_vecs = embed_texts(sent_texts)
    hyp_vecs = embed_texts(hyp_texts)

    results: List[Judgement] = []
    for h, hv in zip(hypotheses, hyp_vecs):
        ev = retrieve_evidence(
            hypothesis=h.hypothesis,
            sentences=sentences,
            sent_vecs=sent_vecs,
            hyp_vec=hv,
            k=k,
            add_neighbors=add_neighbors,
        )
        j = judge_one(h, ev)
        if use_verifier:
            j = verify(j, ev)
        results.append(j)

    entailed_count = sum(1 for r in results if r.entailed)
    total = len(results)
    covered_all = (entailed_count == total) if total > 0 else True
    coverage_rate = (entailed_count / total) if total > 0 else 1.0

    return {
        "covered_all": covered_all,
        "coverage_rate": coverage_rate,
        "entailed": [r.triple_id for r in results if r.entailed],
        "not_entailed": [r.triple_id for r in results if not r.entailed],
        "results": [
            {
                "triple_id": r.triple_id,
                "hypothesis": r.hypothesis,
                "entailed": r.entailed,
                "evidence_sentence_ids": r.evidence_sentence_ids,
                "confidence": r.confidence,
                "reason": r.reason,
            }
            for r in results
        ],
    }


# ----------------------------
# Example usage (you fill in embed_texts + llm_json)
# ----------------------------

if __name__ == "__main__":
    g = Graph()
    # TODO: parse your RDF into g, e.g. g.parse(data=..., format="turtle")
    nt_data="""
    <p1> a <c1> .
    <p1> <http://www.w3.org/2000/01/rdf-schema#label> "Alice" .
    <p1> <age> "30" .
    <p1> <knows> <p2> .
    <p2> a <c1> .
    <p2> <http://www.w3.org/2000/01/rdf-schema#label> "Bob" .
    <c1> <http://www.w3.org/2000/01/rdf-schema#label> "Person" .
    """
    g.parse(data=nt_data, format="ttl")

    text = (
        "Alice is a person. Alice is 30 years old. She knows Bob."
    )

    result = check_coverage(g, text, k=4, ignore_type_triples=True, split_sentences=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    pass
