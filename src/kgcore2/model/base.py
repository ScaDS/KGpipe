# kgcore/model/base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from kgcore.common.types import KGId, Props, new_id

@dataclass
class KGEntity:
    id: KGId = field(default_factory=new_id)
    labels: List[str] = field(default_factory=list)
    props: Props = field(default_factory=dict)

@dataclass
class KGRelation:
    id: KGId = field(default_factory=new_id)
    type: str = ""
    source: KGId = ""
    target: KGId = ""
    props: Props = field(default_factory=dict)

class KGGraph:
    def create_entity(self, labels: List[str], props: Props | None = None, id: KGId = None) -> KGEntity: ...
    def create_relation(self, type: str, source: KGId, target: KGId, props: Props | None = None) -> KGRelation: ...
    def add_meta(self, target: KGId | tuple[KGId, str, KGId], meta: Props) -> None:
        """Attach metadata to an entity, relation, or explicit triple."""
    def find_entities(self, label: str | None = None, props: Props | None = None) -> List[KGEntity]: ...
    def find_relations(self, type: str | None = None, props: Props | None = None) -> List[KGRelation]: ...
    def add_object(self,obj: Any): ...
    def update_entity(self, id: KGId, props: Props) -> None: ...
