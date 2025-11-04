import pandas as pd
import pytest
from pathlib import Path
from rdflib import Graph

from kgpipe.common import Data, DataFormat, KG
from kgpipe.datasets.multipart_multisource import load_dataset
from kgpipe.evaluation.aspects import semantic, statistical

from moviekg.evaluation.helpers import metrics_to_long_table_rows, print_long_table_rows
from moviekg.config import dataset, OUTPUT_ROOT

dataset = dataset
ontology = Graph()
ontology.parse(dataset.ontology)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def test_evaluate_seed_kg():
    
    kg_seed = dataset.splits["split_0"].kg_seed
    if kg_seed is None:
        raise ValueError("KG seed is not found for split 0")
    
    kg = KG(id="seed", name="seed", path=kg_seed.root / "data.nt", format=DataFormat.RDF_NTRIPLES)
    kg.set_ontology_graph(ontology)

    stats = statistical.StatisticalEvaluator()
    stats_results = stats.evaluate(kg)
    
    semac = semantic.SemanticEvaluator()
    semac_results = semac.evaluate(kg)
    
    rows = metrics_to_long_table_rows(stats_results.metrics, "seed", "0")
    rows.extend(metrics_to_long_table_rows(semac_results.metrics, "seed", "0"))
    
    print_long_table_rows(rows)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_ROOT / f"seed_0_metrics.csv", index=False)
    print("saved metrics to", OUTPUT_ROOT / f"seed_0_metrics.csv")

@pytest.mark.parametrize("split_id", range(0, 4))
def test_evaluate_reference_kg(split_id: int):
    print(f"Evaluating reference kg for split {split_id}")

    kg_reference = dataset.splits[f"split_{split_id}"].kg_reference
    if kg_reference is None:
        raise ValueError(f"KG reference is not found for split {split_id}")
    
    kg = KG(id="reference", name="reference", path=kg_reference.root / "data_agg.nt", format=DataFormat.RDF_NTRIPLES)
    kg.set_ontology_graph(ontology)

    stats = statistical.StatisticalEvaluator()
    stats_results = stats.evaluate(kg)
    
    semac = semantic.SemanticEvaluator()
    semac_results = semac.evaluate(kg)
    
    rows = metrics_to_long_table_rows(stats_results.metrics, "reference", "stage_"+str(split_id))
    rows.extend(metrics_to_long_table_rows(semac_results.metrics, "reference", "stage_"+str(split_id)))
    
    print_long_table_rows(rows)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_ROOT / f"reference_{split_id}_metrics.csv", index=False)
    print("saved metrics to", OUTPUT_ROOT / f"reference_{split_id}_metrics.csv")