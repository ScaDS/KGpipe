from dataclasses import dataclass
from kgcore.api.kg import KnowledgeGraph, KGProperty
from kgcore.backend.rdf import RDFLibBackend
from kgcore.model.rdf import RDFBaseModel

model=RDFBaseModel()
backend=RDFLibBackend()
kg = KnowledgeGraph(model=model,backend=backend)

kg.create_entity("http://example.org/explicit/iri", ["foo"], [KGProperty("bar","buz")])

for t in model.triples(backend,None,None,None):
    print(t)

###

@dataclass
class DataInfoDef():
    hash: str
    timestamp: str
    version: str

# src/kgpipe/common/models.py
@dataclass
class DataDef():
    format: str
    path: str
    info: DataInfoDef


# src/kgpipe/common/models.py
class PipelienDef():
    pass

###

# src/kgpipe/common/models.py
class TaskFuntion():
    pass


class EvalFunction():
    pass

### 

class PipelineResult():
    pass

# src/kgpipe/common/models.py
class TaskResult():
    pass

class EvalResult():
    pass

###

class Configuration: pass
class PipelineLayout: pass