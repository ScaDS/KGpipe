from kgpipe_eval.utils.kg_utils import KgManager
from kgpipe_eval.test.utils import get_test_kg, get_reference_kg
from pathlib import Path

tmp_dir = Path("tmp_test_data")

def test_substract_kg():
    # TODO test can be improved / cleaned up
    kg = get_test_kg()
    kg_graph = KgManager.load_kg(kg)
    kg_path = kg.path

    # read kg 
    with open(kg_path, "r") as f:
        triples = f.readlines()
    sample_triples = triples[:10]
    other_kg_path = tmp_dir / "other_kg.nt"
    with open(other_kg_path, "w") as f:
        f.write("\n".join(sample_triples))
    other_kg_graph = KgManager.load_kg(other_kg_path)
    
    substracted_kg_graph = KgManager.substract_kg(kg_graph, other_kg_graph)
    len_kg_triples = len(list(kg_graph.triples((None, None, None))))
    len_other_kg_triples = len(list(other_kg_graph.triples((None, None, None))))
    len_substracted_kg_triples = len(list(substracted_kg_graph.triples((None, None, None))))
    # print(f"len_kg_triples: {len_kg_triples}")
    # print(f"len_other_kg_triples: {len_other_kg_triples}")
    # print(f"len_substracted_kg_triples: {len_substracted_kg_triples}")
    assert len_substracted_kg_triples == len_kg_triples - len_other_kg_triples