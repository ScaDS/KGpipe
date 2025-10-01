from kgpipe.common import Data, DataFormat, Registry


# There are many cleaning tasks
# remove duplicates
# remove nulls
# remove inconsistent triples

@Registry.task({"in": DataFormat.RDF}, {"out": DataFormat.RDF})
def remove_inconsistent_rdf_triples(input: dict[str, Data], output: dict[str, Data]):
    raise NotImplementedError("Not implemented")

@Registry.task({"in": DataFormat.RDF}, {"out": DataFormat.RDF})
def remove_empty_rdf_literals(input: dict[str, Data], output: dict[str, Data]):
    raise NotImplementedError("Not implemented")

