from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Union, runtime_checkable, Optional, Tuple, Literal

from rdflib import RDF, Graph
from rdflib.term import Identifier
from rdflib.term import Literal
from rdflib.term import URIRef

from kgpipe.common import KG


KgLike = Union[KG, Graph, str, Path]

Term = Union[Identifier, str, URIRef, Literal]

Triple = tuple[Term, Term, Term]

TriplePattern = Tuple[
    Optional[Term], Optional[Term], Optional[Term]
]

@runtime_checkable
class TripleGraph(Protocol):
    """
    TripleGraph is a protocol that defines the interface for a graph that can be used to evaluate metrics.
    It is used to abstract the underlying graph implementation and allow for different graph implementations to be used.

    This is intentionally small: metrics should depend on *these* operations,
    not on a specific in-memory representation (RDFLib Graph today; Spark later).
    """

    def triples(self, triple_pattern: TriplePattern) -> Iterable[Triple]:
        pass

    def close(self) -> None:
        pass

    def cache(self) -> None:
        pass

#     def iter_triples(self) -> Iterable[Triple]:
#         """Iterate (s, p, o) triples."""

#     @property
#     def triples(self) -> frozenset[Triple]:
#         """Materialized triple set (may be expensive)."""

#     @property
#     def entities(self) -> frozenset[Term]:
#         """All subjects/objects that are IRIs or blank nodes (no literals)."""

#     @property
#     def relations(self) -> frozenset[Term]:
#         """All predicates."""

#     @property
#     def classes(self) -> frozenset[Term]:
#         """All classes used in rdf:type assertions."""

#     @property
#     def class_occurrences(self) -> Mapping[Term, int]:
#         """Class → number of rdf:type occurrences."""

@dataclass(frozen=True)
class SparkTripleGraph(TripleGraph):
    """
    KG backend that exposes evaluation-friendly views derived from a Spark DataFrame.
    """
    # df: SparkDataFrame

    def triples(self, triple_pattern: TriplePattern) -> Iterable[Triple]:
        # return self.df.filter(triple_pattern).collect()
        pass

    def close(self) -> None:
        pass

    def cache(self) -> None:
        pass

@dataclass(frozen=True)
class RdfLibTripleGraph(TripleGraph):
    """
    KG backend that exposes evaluation-friendly views derived from an RDFLib `Graph`.

    Accepts:
    - `kgpipe.common.KG` (uses `get_graph()`)
    - an RDFLib `Graph`
    - a path/str (parsed by RDFLib)
    """
    kg: KgLike

    def _graph(self) -> Graph:
        if isinstance(self.kg, Graph):
            return self.kg
        if isinstance(self.kg, KG):
            return self.kg.get_graph()
        # Assume filesystem path
        return Graph().parse(str(self.kg))

    def triples(self, triple_pattern: TriplePattern) -> Iterable[Triple]:
        g = self._graph()
        # RDFLib yields (s, p, o) as Identifiers
        return g.triples(triple_pattern)

class KgManager:
    """
    KgManager is a class that manages the loading and unloading of KGs.
    It is used to abstract the underlying graph implementation and allow for different graph implementations to be used.
    """

    @staticmethod
    def load_kg(kg: KG, backend: Literal["rdflib", "spark"] = "rdflib") -> TripleGraph:
        if backend == "rdflib":
            return RdfLibTripleGraph(kg=kg)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def load_kg_from_path(path: Path, backend: Literal["rdflib", "spark"] = "rdflib") -> TripleGraph:
        if backend == "rdflib":
            return RdfLibTripleGraph(kg=path)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def cache_kg(kg: TripleGraph) -> None:
        kg.cache()

    @staticmethod
    def unload_kg(kg: TripleGraph) -> None:
        kg.close()
