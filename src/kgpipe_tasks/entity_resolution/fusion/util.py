from typing import Optional
from kgpipe.evaluation.cluster import MatchCluster
from kgpipe_tasks.transform_interop.exchange.entity_matching import ER_Document
import json

def load_matches_from_file(file_path, threshold, type_filter: Optional[str] = None) -> MatchCluster:

    er = ER_Document(**json.load(open(file_path)))

    matches = MatchCluster()

    for match in er.matches:
        if match.score > threshold:
            if type_filter and match.id_type != type_filter:
                continue
            id_1 = str(match.id_1) # .split("-")[0] # paris workaround
            id_2 = str(match.id_2) # .split("-")[0] # paris workaround
            if id_1.endswith("-") or id_2.endswith("-"):
                continue
            if id_1.endswith("_uri") or id_1.endswith("_literal"):
                id_1 = id_1.replace("_uri", "").replace("_literal", "")
            if id_2.endswith("_uri") or id_2.endswith("_literal"):
                id_2 = id_2.replace("_uri", "").replace("_literal", "")
            # if match.id_type == "relation":
                # print(f"Adding relation match: {id_1} {id_2}")
            matches.add_match(id_1, id_2)

    return matches