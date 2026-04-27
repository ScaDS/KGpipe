import json
import os
from logging import getLogger
from pathlib import Path
from typing import Dict, List

from kgcore.api.ontology import OntologyUtil
from kgpipe.common.config import TARGET_ONTOLOGY_NAMESPACE
from kgpipe.common.model.configuration import ConfigurationDefinition
from kgpipe.common.models import Data, DataFormat, KgTask
from pydantic import BaseModel
from rdflib import Graph, RDF, RDFS, SKOS, URIRef

logger = getLogger(__name__)

class TrackRecord(BaseModel):
    original_subject: str
    subject: str
    original_predicate: str
    predicate: str
    original_object: str
    object: str


def select_first_value_function(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    """
    For two KGs A and B, merge A into B where for each s_p and 
    1) p is fusable and B does not have any s_p_o or 
    2) p is not fusable erge all s_p_o
    """
    ontology_path = os.environ.get("ONTOLOGY_PATH", "false")
    if ontology_path == "false":
        raise ValueError("ONTOLOGY_PATH is not set")

    ontology = OntologyUtil.load_ontology_from_file(Path(ontology_path))
    allowed_predicates = set[str]([str(p.uri) for p in ontology.properties]+[str(RDFS.label), str(RDF.type), str(SKOS.altLabel)])
    fusable_properties = set[str]([str(p.uri) for p in ontology.properties if p.max_cardinality == 1]+[str(RDFS.label), str(RDF.type)])  

    def is_fusable(p):
        return str(p) in fusable_properties

    source_graph = Graph()
    source_graph.parse(inputs["source"].path, format="nt")
    seed_graph = Graph() # seed graph
    seed_graph.parse(inputs["target"].path, format="nt")

    current_subjects = set[str]([str(s) for s in seed_graph.subjects(unique=True)])

    selected: List[TrackRecord] = []
    discarded: List[TrackRecord] = []
    
    for s, p, o in source_graph:
        s_can = s
        p_can = p
        o_can = o 

        if not isinstance(p_can, URIRef) or str(p_can) not in allowed_predicates:
            continue
        
        if p_can == RDF.type and not str(o_can).startswith(TARGET_ONTOLOGY_NAMESPACE):
            continue

        if is_fusable(p_can):
            # Add exactly one value if none exists yet
            if not any(seed_graph.objects(s_can, p_can)):
                seed_graph.add((s_can, p_can, o_can))
                selected.append(
                    TrackRecord(
                        subject=str(s_can),
                        predicate=str(p_can),
                        object=str(o_can),
                        original_subject=str(s),
                        original_predicate=str(p),
                        original_object=str(o),
                    )
                )
                # keep subjects set fresh for subsequent matches
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))
            else:
                discarded.append(
                    TrackRecord(
                        subject=str(s_can),
                        predicate=str(p_can),
                        object=str(o_can),
                        original_subject=str(s),
                        original_predicate=str(p),
                        original_object=str(o),
                    )
                )
        else:
            # Non-fusable: copy if not already present (avoid dupes)
            if (s_can, p_can, o_can) not in seed_graph:
                seed_graph.add((s_can, p_can, o_can))
                if isinstance(s_can, URIRef):
                    current_subjects.add(str(s_can))

    # sel(ected)
    selected_file_path = outputs["output"].path.parent / (outputs["output"].path.stem + ".selected.json")
    with open(selected_file_path, "w") as f:
        json.dump(selected, f, default=lambda x: x.model_dump())
    # dis(carded)
    discarded_file_path = outputs["output"].path.parent / (outputs["output"].path.stem + ".discarded.json")
    with open(discarded_file_path, "w") as f:
        json.dump(discarded, f, default=lambda x: x.model_dump())

    # prov graph is skipped here as no uris are replaced (is done in previouse steps)
    seed_graph.serialize(outputs["output"].path, format="nt")

select_first_value_task = KgTask(
    name="select_first_value",
    input_spec={"source": DataFormat.RDF_NTRIPLES, "target": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    function=select_first_value_function,
    config_spec=ConfigurationDefinition(
        name="select_first_value",
        parameters=[]
    )
)