from pathlib import Path
from kgpipe.common import Registry, KG, Data, DataFormat
from kgpipe_llm.common.core import get_client_from_env
from kgpipe_llm.common.snippets import generate_ontology_snippet
from kgpipe_llm.common.models import OntologyMappings
from kgpipe_tasks.common.ontology import Ontology, OntologyUtil
from pydantic import BaseModel
from typing import Optional, Dict, Callable
import jsonpath_ng
import json
import os
from rdflib import Graph, RDF
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document, ER_Match


from collections import defaultdict, Counter
import random
from typing import Iterable, Tuple, Any, Dict, List, Set

Triple = Tuple[Any, Any, Any]

class SubgraphSamplerWithNthRelations:
    @staticmethod
    def create_sample_entities(graph, samples_n: int, *, seed: int = 42) -> Any:
        """
        Build a small, closed subgraph that:
          1) contains at least `samples_n` triples for *each* predicate in `graph`,
          2) minimizes the number of distinct entities (subjects/objects) by
             aggressively preferring overlapping nodes across predicates, and
          3) is *closed*: no dangling entities — every object in the subgraph
             must also appear as a subject in the subgraph.

        Notes
        -----
        - This is a greedy heuristic (the exact problem is NP-hard).
        - Works with rdflib-like Graphs (iterable of (s, p, o) and `.add((s,p,o))`).
        - If a chosen triple would make closure impossible (object with no
          outgoing triples), we backtrack on that triple and try alternatives.
        - If a predicate has fewer than `samples_n` triples in the original
          graph, we include all available.
        """
        rng = random.Random(seed)

        # --- Index the graph ---
        all_triples: List[Triple] = []
        triples_by_pred: Dict[Any, List[Triple]] = defaultdict(list)
        outgoing_by_subj: Dict[Any, List[Triple]] = defaultdict(list)
        incoming_by_obj: Dict[Any, List[Triple]] = defaultdict(list)

        for s, p, o in graph:
            t = (s, p, o)
            all_triples.append(t)
            triples_by_pred[p].append(t)
            outgoing_by_subj[s].append(t)
            incoming_by_obj[o].append(t)

        predicates: List[Any] = list(triples_by_pred.keys())
        if not predicates:
            new_graph = type(graph)()
            return new_graph

        # Shuffle within predicates to diversify tie-breaking but keep reproducible
        for p in predicates:
            rng.shuffle(triples_by_pred[p])

        # --- Helpers ---
        def degree(n: Any) -> int:
            return len(outgoing_by_subj.get(n, ())) + len(incoming_by_obj.get(n, ()))

        def candidate_score(t: Triple, selected_entities: Set[Any]) -> Tuple[int, int, int]:
            """
            Score higher for overlap with current entities and for high-degree nodes.
            Returns a tuple where *larger* is better in lexicographic order.
            (overlap_count, degree_sum, -new_entities_count)
            """
            s, _, o = t
            overlap = int(s in selected_entities) + int(o in selected_entities)
            deg_sum = degree(s) + degree(o)
            new_count = (0 if s in selected_entities else 1) + (0 if o in selected_entities else 1)
            return (overlap, deg_sum, -new_count)

        def pick_best(cands: Iterable[Triple], selected_entities: Set[Any]) -> Triple | None:
            best = None
            best_score = None
            for t in cands:
                sc = candidate_score(t, selected_entities)
                if best_score is None or sc > best_score:
                    best = t
                    best_score = sc
            return best

        # --- Greedy selection per predicate (maximize overlap) ---
        selected_triples: Set[Triple] = set()
        selected_entities: Set[Any] = set()
        per_pred_counts: Counter = Counter()

        # Seed with the densest triple overall (helps connectivity)
        seed_candidates = (t for ts in triples_by_pred.values() for t in ts)
        seed_t = pick_best(seed_candidates, set())
        if seed_t is not None:
            selected_triples.add(seed_t)
            s0, p0, o0 = seed_t
            selected_entities.update([s0, o0])
            per_pred_counts[p0] += 1

        # Now ensure quota for each predicate
        for p in predicates:
            if per_pred_counts[p] >= samples_n:
                continue
            needed = samples_n - per_pred_counts[p]

            # Prefer candidates not yet selected and that maximize overlap
            pool = [t for t in triples_by_pred[p] if t not in selected_triples]
            while needed > 0 and pool:
                t = pick_best(pool, selected_entities)
                if t is None:
                    break
                selected_triples.add(t)
                s, _, o = t
                selected_entities.update([s, o])
                per_pred_counts[p] += 1
                needed -= 1
                # Remove t from pool
                pool.remove(t)

        # --- Ensure closure: every object appears as a subject ---
        def compute_dangling(st: Set[Triple]) -> Set[Any]:
            subs = {s for s, _, _ in st}
            objs = {o for _, _, o in st}
            return {o for o in objs if o not in subs}

        def try_fix_node(d: Any) -> bool:
            """Try to fix a dangling node by adding an outgoing triple from d.
            Prefer triples that a) satisfy still-missing predicate quotas, then
            b) maximize overlap. Returns True if fixed, False otherwise."""
            outs = outgoing_by_subj.get(d, [])
            if not outs:
                return False
            # Split by whether predicate still needs quota
            needers = [t for t in outs if per_pred_counts[t[1]] < samples_n and t not in selected_triples]
            others  = [t for t in outs if t not in selected_triples]
            cands = needers if needers else others
            if not cands:
                return False
            t = pick_best(cands, selected_entities)
            if t is None:
                return False
            selected_triples.add(t)
            s, p, o = t
            selected_entities.update([s, o])
            per_pred_counts[p] += 1
            return True

        # Iterate closure until stable; if an unfixable dangling node exists,
        # drop the offending incoming triples and re-fill quotas where needed.
        safety = 0
        while True:
            safety += 1
            if safety > 10000:
                break  # extreme safety guard
            dangling = compute_dangling(selected_triples)
            if not dangling:
                break
            progressed = False
            for d in list(dangling):
                if try_fix_node(d):
                    progressed = True
                    continue
                # Cannot fix by adding; backtrack: remove all triples that point to d
                to_remove = [t for t in selected_triples if t[2] == d]
                if not to_remove:
                    continue
                for t in to_remove:
                    selected_triples.remove(t)
                    per_pred_counts[t[1]] -= 1
                # Re-fill quotas for any predicates that dropped below requirement
                for p in predicates:
                    while per_pred_counts[p] < samples_n:
                        # pick best remaining candidate for this predicate
                        pool = [t for t in triples_by_pred[p] if t not in selected_triples]
                        if not pool:
                            break
                        tbest = pick_best(pool, selected_entities)
                        if tbest is None:
                            break
                        selected_triples.add(tbest)
                        selected_entities.update([tbest[0], tbest[2]])
                        per_pred_counts[p] += 1
                        progressed = True
            if not progressed:
                # No way to progress: break to avoid infinite loop
                break

        # Final pass: if any predicate still below quota (due to extreme pruning), fill
        for p in predicates:
            while per_pred_counts[p] < samples_n:
                pool = [t for t in triples_by_pred[p] if t not in selected_triples]
                if not pool:
                    break
                tbest = pick_best(pool, selected_entities)
                if tbest is None:
                    break
                selected_triples.add(tbest)
                selected_entities.update([tbest[0], tbest[2]])
                per_pred_counts[p] += 1

        # Return as a new graph instance of same type
        new_graph = type(graph)()
        for t in selected_triples:
            new_graph.add(t)
        return new_graph

class SubgraphSamplerWithNthEntities:
    """
    Simpler subgraph sampler that samples n entities from the graph.
    Try to reproduce the same distribution of classes as in the original graph.
    """

    @staticmethod
    def create_sample_entities(graph: Graph, samples_n: int) -> Graph:
        """
        Samples n entities (subjects) from the graph, attempting to match the
        class distribution (rdf:type) of the original graph.

        Strategy:
        1) Compute class frequencies from rdf:type triples.
        2) Compute per-class target counts proportional to frequencies.
        3) Greedily pick subjects that cover the most unmet class demand
           (helpful when subjects have multiple classes).
        4) If needed, fill remaining slots with arbitrary subjects.
        5) Build a subgraph containing all triples with each selected subject as subject.
        """
        new_graph = Graph()

        # Early exits
        if samples_n <= 0 or len(graph) == 0:
            return new_graph

        # All unique subjects in the graph
        subjects = list(graph.subjects(unique=True))
        if not subjects:
            return new_graph

        # Cap samples to available subjects
        samples_n = min(samples_n, len(subjects))

        # Build: subject -> set(classes), and class -> list(subjects)
        subj_to_classes = defaultdict(set)
        class_to_subjs = defaultdict(list)

        for s, _, c in graph.triples((None, RDF.type, None)):
            subj_to_classes[s].add(c)

        for s in subjects:
            for c in subj_to_classes.get(s, ()):
                class_to_subjs[c].append(s)

        # If there are no rdf:type triples, just pick any subjects
        if not class_to_subjs:
            selected = set(subjects[:samples_n])
            for s in selected:
                for t in graph.triples((s, None, None)):
                    new_graph.add(t)
            return new_graph

        # Class frequencies from rdf:type triples (counts (s, rdf:type, c) occurrences)
        class_counts = Counter()
        for c, subjs in class_to_subjs.items():
            class_counts[c] = len(subjs)  # count subjects per class (more stable than raw triple freq)

        total = sum(class_counts.values())
        if total == 0:
            # Fallback (shouldn't happen if we hit the class_to_subjs branch)
            selected = set(subjects[:samples_n])
            for s in selected:
                for t in graph.triples((s, None, None)):
                    new_graph.add(t)
            return new_graph

        # Compute proportional targets per class
        # Use "largest remainder" method to ensure sum == samples_n
        raw_targets = {c: (class_counts[c] / total) * samples_n for c in class_counts}
        targets = {c: int(v) for c, v in raw_targets.items()}
        # Distribute remaining by largest fractional remainder
        remainder = samples_n - sum(targets.values())
        if remainder > 0:
            remainders_sorted = sorted(
                class_counts.keys(),
                key=lambda c: (raw_targets[c] - targets[c]),
                reverse=True
            )
            for c in remainders_sorted[:remainder]:
                targets[c] += 1

        # Greedy selection: pick subjects that cover the most unmet class demand
        selected = set()
        deficits = targets.copy()

        # Build a pool of candidate subjects that have at least one class we still need
        # (we’ll refresh this inside the loop as deficits change)
        def subject_gain(s):
            return sum(1 for c in subj_to_classes.get(s, ()) if deficits.get(c, 0) > 0)

        while len(selected) < samples_n and any(v > 0 for v in deficits.values()):
            # Among unselected subjects, pick the one covering most unmet classes
            best_s = None
            best_gain = 0
            for s in subjects:
                if s in selected:
                    continue
                g = subject_gain(s)
                if g > best_gain:
                    best_gain = g
                    best_s = s

            # If no subject improves deficits, break out
            if best_s is None or best_gain == 0:
                break

            selected.add(best_s)
            # Decrease deficits for the classes this subject satisfies
            for c in subj_to_classes.get(best_s, ()):
                if deficits.get(c, 0) > 0:
                    deficits[c] -= 1

        # If we still need more subjects (e.g., overlap/coverage limits), fill arbitrarily
        if len(selected) < samples_n:
            for s in subjects:
                if s not in selected:
                    selected.add(s)
                    if len(selected) >= samples_n:
                        break

        # Build the sampled subgraph: include all triples where selected subject is the subject
        for s in selected:
            for t in graph.triples((s, None, None)):
                new_graph.add(t)

        return new_graph

class Match_Pair_v1(BaseModel):
    """
    A match between two entities
    """
    id_1: str 
    id_2: str

class Match_Pairs_v1(BaseModel):
    matched_relations: List[Match_Pair_v1]
    # matched_classes: List[Match_Pair_v1]

# "matched_classes": [ 
#     // list of pairs: ["class_in_target", "class_in_input"]
# ]
# - Find **matching classes** (rdf:type or class definitions).  
# - Each **input class** may map to **one target class**, and multiple input classes may map to the same target class. 

class RDF_OM_v1:

    @staticmethod
    def create_sample_entities(graph: Graph, samples_n: int):
        """
        Reduces the graph to a collection of entites, covering all relations in the graph.
        """
        subgraph = SubgraphSamplerWithNthEntities.create_sample_entities(graph, samples_n)
        return subgraph

    @staticmethod
    def match_relations(subgraph_input: Graph, subgraph_target: Graph) -> Match_Pairs_v1:
        """
        Matches the relations in the input subgraph to the relations in the target subgraph.
        """
        
        prompt = f"""
You are given two RDF subgraphs in Turtle format: a *target* subgraph and an *input* subgraph.  
Your task is to identify correspondences between them.

Rules for matching:
- Find **matching relations** (predicates/properties).  
- Each **input relation** (predicate/property) may map to **one target relation**, and multiple input relations may map to the same target relation.  
- Do **not** produce n-to-m matches (i.e., one input item cannot map to multiple targets).  

Return your answer **only** as a JSON object with the following structure:

{{
"matched_relations": [ 
    // list of pairs: ["relation_in_input", "relation_in_target"]
]
}}

Do not include explanations or text outside the JSON.

---

### Target Subgraph (Turtle)
{subgraph_target.serialize(format="turtle")}

### Input Subgraph (Turtle)
{subgraph_input.serialize(format="turtle")}
        """

        system_prompt = "You are a expert in RDF ontology matching."

        response_data = get_client_from_env().send_prompt(prompt,Match_Pairs_v1,system_prompt)
        
        try:
            return Match_Pairs_v1(**response_data)
        except Exception as e:
            print(f"Error matching relations: {e}")
            return Match_Pairs_v1(matched_relations=[]) # , matched_classes=[])

    @staticmethod
    def rdf_om_from_files_to_er_json_file(input_path: Path, target_path: Path, output_path: Path) -> None:
        input_graph = Graph().parse(input_path, format="turtle")
        target_graph = Graph().parse(target_path, format="turtle")
        subgraph_input = RDF_OM_v1.create_sample_entities(input_graph, 50)
        subgraph_target = RDF_OM_v1.create_sample_entities(target_graph, 50)

        matched_relations = RDF_OM_v1.match_relations(subgraph_input, subgraph_target)

        er_pairs = []
        for relation in matched_relations.matched_relations:
            er_pairs.append(ER_Match(id_1=relation.id_1, id_2=relation.id_2, score=1, id_type="relation"))
        # for class_ in matched_relations.matched_classes:
        #     er_pairs.append(ER_Match(id_1=class_.id_1, id_2=class_.id_2, score=1, id_type="entity"))
        er_document = ER_Document(matches=er_pairs)

        with open(output_path, "w") as f:
            f.write(er_document.model_dump_json(indent=2))

@Registry.task(
    description="RDF Ontology Matching",
    input_spec={"input": DataFormat.RDF_NTRIPLES, "kg": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.ER_JSON},
    category=["RDF", "Ontology Matching"]
)
def llm_task_rdf_ontology_matching_v1(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    input_path = inputs["input"].path
    target_path = inputs["kg"].path
    output_path = outputs["output"].path

    RDF_OM_v1.rdf_om_from_files_to_er_json_file(input_path, target_path, output_path)


@Registry.task(
    description="Map ER Match Relations",
    input_spec={"matches": DataFormat.ER_JSON, "input": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    category=["RDF", "Ontology Matching"]
)
def map_er_match_relations(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    matches_path = inputs["matches"].path
    input_path = inputs["input"].path
    output_path = outputs["output"].path

    er_document = ER_Document.model_validate_json(open(matches_path).read())
    input_graph = Graph().parse(input_path)

    new_graph = Graph()

    matches = {}
    matches_inv = {}

    for match in er_document.matches:
        matches[match.id_1] = match.id_2
        matches_inv[match.id_2] = match.id_1

    for s, p, o in input_graph:
        if p in matches:
            new_graph.add((s, p, matches[p]))
        elif p in matches_inv:
            new_graph.add((matches_inv[p], p, o))
        else:
            new_graph.add((s, p, o))

    new_graph.serialize(output_path, format="nt")