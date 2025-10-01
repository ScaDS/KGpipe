from kgpipe_tasks.common.ontology import Ontology, OntologyUtil
from pathlib import Path

ontology = OntologyUtil.load_ontology_from_file(Path("/home/marvin/project/code/experiments/movie-ontology.ttl"))

for prop in ontology.properties:
    print(prop.uri)
    print(prop.domain.uri)
    print(prop.range.uri)
    print("---")