from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from rdflib import Graph, URIRef, Literal, RDF, RDFS, XSD
import re
from tqdm import tqdm

def integrated_entities(path_actual_kg, path_expected_kg):
    pass

SOFT_ENTITY_THRESHOLD = 0.75
SOFT_VALUES_THRESHOLD = 0.75

def encode(values, model, desc: str):
    embeddings = []
    for i in tqdm(range(0, len(values), 64), desc=desc):
        batch = values[i:i+64]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

def graph_fact_alginment(ga: Graph, ge: Graph):
    te = [ str(s)+str(p)+str(o) for s, p, o in ge ]
    ta = [ str(s)+str(p)+str(o) for s, p, o in ga ]

    tp = len(set(ta) & set(te))
    fp = len(set(ta) - set(te))
    fn = len(set(te) - set(ta))

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"F1: {2 * tp / (2 * tp + fp + fn)}")

def clean_label(label: str):
    # remove all non-alphanumeric characters
    cleaned_label = label.replace("_", " ")
    # remove parenthesis text
    cleaned_label = re.sub(r'\([^)]*\)', '', cleaned_label)
    return cleaned_label.strip()


def graph_match_labels_soft(ga: Graph, ge: Graph, model: SentenceTransformer):
    actual_uri_to_abels = {}
    expected_uri_to_abels = {}

    for s, _, o in ga.triples((None, RDFS.label, None)):
        actual_uri_to_abels[str(s)] = clean_label(str(o))

    for s, _, o in ge.triples((None, RDFS.label, None)):
        expected_uri_to_abels[str(s)]= clean_label(str(o))

    actual_embeddings = encode(list(actual_uri_to_abels.values()), model, "Encoding actual labels")
    expected_embeddings = encode(list(expected_uri_to_abels.values()), model, "Encoding expected labels")

    cosine_scores = np.dot(actual_embeddings, expected_embeddings.T)

    actual_uri_keys = list(actual_uri_to_abels.keys())
    expected_uri_keys = list(expected_uri_to_abels.keys())

    # get best match expected uri for each actual uri
     
    uri_mappings = {}

    best_matches = []
    for i in range(len(actual_uri_keys)):
        best_match = expected_uri_keys[np.argmax(cosine_scores[i])]
        best_score = cosine_scores[i][np.argmax(cosine_scores[i])]
        best_matches.append((best_match, best_score))

    for i in range(len(best_matches)):
        if best_matches[i][1] > SOFT_ENTITY_THRESHOLD:
            # la = actual_uri_to_abels[actual_uri_keys[i]].replace(" ", "_")
            # le = expected_uri_to_abels[best_matches[i][0]].replace(" ", "_")
            uri_actual = actual_uri_keys[i]
            uri_expected = best_matches[i][0]
            uri_mappings[uri_actual] = uri_expected

    return uri_mappings

def graph_fact_alginment_soft_entities(ga: Graph, ge: Graph, model: SentenceTransformer):
    uri_mappings = graph_match_labels_soft(ga, ge, model)
    
    ga_mapped = Graph()
    for s, p, o in ga:
        if str(s) in uri_mappings:
            s = URIRef(uri_mappings[str(s)])
        if isinstance(o, URIRef) and str(o) in uri_mappings:
            o = URIRef(uri_mappings[str(o)])
        ga_mapped.add((s, p, o))

    graph_fact_alginment(ga_mapped, ge)

# TODO rdf:type is removed for tp calculation
def graph_fact_alginment_soft_entities_values(ga: Graph, ge: Graph, model: SentenceTransformer):
    uri_mappings = graph_match_labels_soft(ga, ge, model)

    def get_label(o: URIRef, graph: Graph):
        labels = [str(l) for l in graph.objects(o, RDFS.label)]
        if len(labels) == 0:
            return []
        else:
            return [clean_label(l) for l in labels]
    
    ga_mapped = Graph()
    for s, p, o in ga:
        if str(s) in uri_mappings:
            s = URIRef(uri_mappings[str(s)])
        if isinstance(o, URIRef): # and p != RDF.type
            for label in get_label(o, ga):
                ga_mapped.add((s, p, Literal(label)))
        else:
            ga_mapped.add((s, p, o))

    ge_mapped = Graph()
    for s, p, o in ge:
        if isinstance(o, URIRef): # and p != RDF.type
            for label in get_label(o, ge):
                ge_mapped.add((s, p, Literal(label)))
        else:
            ge_mapped.add((s, p, o))

    # encode all values
    vas = list(set([str(o) for _, _, o in ga_mapped if not isinstance(o, URIRef)]))
    ves = list(set([str(o) for _, _, o in ge_mapped if not isinstance(o, URIRef)]))

    va_embeddings = encode(vas, model, "Encoding actual values")
    ve_embeddings = encode(ves, model, "Encoding expected values")

    v2e_actual = {}
    v2e_expected = {}

    for idx, v in enumerate(vas):
        v2e_actual[v] = va_embeddings[idx]

    for idx, v in enumerate(ves):
        v2e_expected[v] = ve_embeddings[idx]

    tp = 0
    fp = 0
    fn = 0

    sp_actual = set()
    
    # for each (s, p, o) in ga_mapped check if there is a matching value for the same (s, p) in ge
    for s, p in ga_mapped.subject_predicates(unique=True):
        sp_actual.add((s, p))
        _vas = [str(o) for o in ga_mapped.objects(s, p)]
        _ves = [str(o) for o in ge_mapped.objects(s, p)]
        _vas_embeddings = np.array([v2e_actual[v] for v in _vas])
        _ves_embeddings = np.array([v2e_expected[v] for v in _ves])

        if len(_vas_embeddings) == 0 or len(_ves_embeddings) == 0:
            continue
        cosine_scores = np.dot(_vas_embeddings, _ves_embeddings.T) # (len(_vas_embeddings), len(_ves_embeddings))

        for idx in range(len(_vas)):
            best_match = _ves[np.argmax(cosine_scores[idx])]
            best_score = cosine_scores[idx][np.argmax(cosine_scores[idx])]
            if best_score > SOFT_VALUES_THRESHOLD:
                actual_value = _vas[idx]
                reference_value = best_match
                tp += 1
                # if actual_value == reference_value:
                #     # print(f"Found matching value for {s} {p} {actual_value}")
                #     pass
                # else:
                #     print(f"Found matching value for {s} {p} {actual_value} but not exact reference {reference_value}")
                # print(f"Value actual: {_vas[idx]}, {best_match}, {best_score}")
                # print(f"Value expected: {_ves[np.argmax(cosine_scores[idx])]}")
            else:
                fp += 1
                # print(f"No matching value for {s} {p} {_vas[idx]} from references {_ves}")
    
    sp_expected = set([(s, p) for s, p in ge_mapped.subject_predicates(unique=True)])
    missing_sp = sp_expected - sp_actual
    for s, p in missing_sp:
        for _ in ge_mapped.triples((s, p, None)):
            fn += 1

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"F1: {2 * tp / (2 * tp + fp + fn)}")

def reference_alignment(path_actual_kg: Path, path_expected_kg: Path):
    ga = Graph()
    ga.parse(path_actual_kg)

    ge = Graph()
    ge.parse(path_expected_kg)

    graph_fact_alginment(ga, ge)

def reference_alignment_soft_entities(path_actual_kg: Path, path_expected_kg: Path):

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to("cuda")

    ga = Graph()
    ga.parse(path_actual_kg)

    ge = Graph()
    ge.parse(path_expected_kg)
    
    graph_fact_alginment_soft_entities(ga, ge, model)

def reference_alignment_soft_entities_values(path_actual_kg: Path, path_expected_kg: Path):

    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.to("cuda")

    ga = Graph()
    ga.parse(path_actual_kg)

    ge = Graph()
    ge.parse(path_expected_kg)
    
    graph_fact_alginment_soft_entities_values(ga, ge, model)

def test_integrated_verified_source_entities():
    print("Integrated verified source entities")
    path_actual_kg = Path("/home/marvin/project/code/experiments/out_film_100/rdf_a/stage_1/result.nt")
    path_expected_kg = Path("/home/marvin/project/data/final/film_100/split_3/kg/reference/data_agg.nt")
    integrated_entities(path_actual_kg, path_expected_kg)

def test_reference_alignment():
    print("Reference alignment")
    path_actual_kg = Path("/home/marvin/project/code/experiments/out_film_100/rdf_a/stage_1/result.nt")
    path_expected_kg = Path("/home/marvin/project/data/final/film_100/split_3/kg/reference/data_agg.nt")
    reference_alignment(path_actual_kg, path_expected_kg)

def test_reference_alignment_soft():
    print("Reference alignment soft")
    path_actual_kg = Path("/home/marvin/project/code/experiments/out_film_100/text_b/stage_1/result.nt")
    path_expected_kg = Path("/home/marvin/project/data/final/film_100/split_3/kg/reference/data_agg.nt")
    reference_alignment_soft_entities(path_actual_kg, path_expected_kg)

def test_reference_alignment_soft_entities_values():
    print("Reference alignment soft entities values")
    path_actual_kg = Path("/home/marvin/project/code/experiments/out_film_100/rdf_a/stage_1/result.nt")
    path_expected_kg = Path("/home/marvin/project/data/final/film_100/split_3/kg/reference/data_agg.nt")
    reference_alignment_soft_entities_values(path_actual_kg, path_expected_kg)

if __name__ == "__main__":
    test_integrated_verified_source_entities()
    test_reference_alignment()