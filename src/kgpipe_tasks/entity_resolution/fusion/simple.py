from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
from kgpipe.common.models import  DataFormat, Data
from typing import Dict
import json
from kgpipe.common.registry import Registry
from logging import getLogger

logger = getLogger(__name__)

@Registry.task(
    input_spec={"json1": DataFormat.ER_JSON, "json2": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.ER_JSON},
    description="Fuse RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def aggregate_2matches(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    er1 = ER_Document(**json.load(open(inputs["json1"].path)))
    er2 = ER_Document(**json.load(open(inputs["json2"].path)))
    
    n_er1 = []
    for er in er1.matches:
        if er.id_type:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, id_type=er.id_type.replace("str","relation"), score=er.score)
        else:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, score=er.score)
        n_er1.append(ne)
    n_er2 = []
    for er in er2.matches:
        if er.id_type:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, id_type=er.id_type.replace("str","relation"), score=er.score)
        else:
            ne = ER_Match(id_1=er.id_1, id_2=er.id_2, score=er.score)
        n_er2.append(ne)
    er_comb = ER_Document(matches=n_er1 + n_er2)
    
    with open(outputs["output"].path, "w") as f:
        json.dump(er_comb.model_dump(), f, indent=4)

from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Match

@Registry.task(
    input_spec={"json1": DataFormat.ER_JSON},
    output_spec={"output": DataFormat.ER_JSON},
    description="Fuse RDF entities using first value fusion",
    category=["EntityResolution", "Fusion"]
)
def reduce_to_best_match_per_entity(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    er1 = ER_Document(**json.load(open(inputs["json1"].path)))

    # Sort all candidate matches by score (high to low)
    sorted_matches = sorted(er1.matches, key=lambda m: m.score, reverse=True)

    selected_matches = []
    matched_ids = set()  # entities already committed to a match

    for match in sorted_matches:
        id1 = match.id_1
        id2 = match.id_2
        # Only accept if both endpoints are currently unmatched
        if id1 not in matched_ids and id2 not in matched_ids:
            selected_matches.append(match)
            matched_ids.add(id1)
            matched_ids.add(id2)

    er_comb = ER_Document(matches=selected_matches)
    with open(outputs["output"].path, "w") as f:
        json.dump(er_comb.model_dump(), f, indent=4)