# kgcore/backend/memory.py
from __future__ import annotations
from typing import List, Dict
from kgcore.model.base import KGGraph, KGEntity, KGRelation
from kgcore.common.types import Props, KGId

class InMemoryGraph(KGGraph):
    def __init__(self) -> None:
        self.entities: Dict[KGId, KGEntity] = {}
        self.relations: Dict[KGId, KGRelation] = {}
        self.meta: Dict[KGId, Props] = {}

    def create_entity(self, labels: List[str], props: Props | None = None, id: KGId = None) -> KGEntity:
        if id:
            e = KGEntity(id=id, labels=list(labels or []), props=dict(props or {}))
        else:
            e = KGEntity(labels=list(labels or []), props=dict(props or {}))
        self.entities[e.id] = e
        return e

    def create_relation(self, type: str, source: KGId, target: KGId, props: Props | None = None) -> KGRelation:
        r = KGRelation(type=type, source=source, target=target, props=dict(props or {}))
        self.relations[r.id] = r
        return r

    def add_meta(self, target: KGId, meta: Props) -> None:
        self.meta[target] = {**self.meta.get(target, {}), **meta}

    def _match_props(self, have: Props, want: Props) -> bool:
        return all(have.get(k) == v for k, v in want.items())

    def find_entities(self, label: str | None = None, props: Props | None = None) -> List[KGEntity]:
        out = self.entities.values()
        if label: out = [e for e in out if label in e.labels]
        if props: out = [e for e in out if self._match_props(e.props, props)]
        return list(out)

    def find_relations(self, type: str | None = None, props: Props | None = None) -> List[KGRelation]:
        out = self.relations.values()
        if type: out = [r for r in out if r.type == type]
        if props: out = [r for r in out if self._match_props(r.props, props)]
        return list(out)

    def update_entity(self, id: KGId, props: Props) -> None:
        self.entities[id].props = {**self.entities[id].props, **props}