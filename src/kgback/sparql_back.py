
from ast import Dict
import re
from traceback import print_tb
from kgcore.model.base import KGGraph
from kgcore.common.types import Props, KGId, Lit, new_id, is_uri
from kgcore.model.base import KGEntity, KGRelation
from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
from typing import List, Mapping, Iterable
import urllib.parse
from pydantic import BaseModel
from rdflib import Literal, URIRef, Graph, RDF
import requests
from typing import Any
from requests.auth import HTTPDigestAuth

class SparqlGraph(KGGraph):
    def __init__(self, endpoint: str, base_iri: str = "http://example.org/", 
                 username: str = None, password: str = None, auth_type: str = "digest"):
        self.endpoint = endpoint
        self.base_iri = base_iri.rstrip('/') + '/'
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        
        # Store authentication credentials for use with requests
        self.username = username
        self.password = password
        self.auth_type = auth_type.lower() if auth_type else None
        
        # Configure SPARQLWrapper for non-authenticated requests
        if not username or not password:
            # No authentication needed
            pass
        elif auth_type.lower() == "basic":
            self.sparql.setCredentials(username, password)
        else:
            # For digest auth, we'll use requests directly instead of SPARQLWrapper
            pass

        try:
            self._execute_select("SELECT * { ?s ?p ?o } LIMIT 1")
        except Exception as e:
            print(f"Error querying SPARQL endpoint: {e}")
            raise Exception(f"Error querying SPARQL endpoint: {e}")
    
    def _encode_basic_auth(self, username: str, password: str) -> str:
        """Encode credentials for basic authentication."""
        import base64
        credentials = f"{username}:{password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return encoded

    def _get_uri(self, id: KGId) -> str:
        """Convert KGId to full URI using base_iri."""
        if id.startswith('http://') or id.startswith('https://'):
            return id
        return self.base_iri + id

    def _serialize_value(self, value) -> str:
        """Serialize a value for SPARQL, handling Literal objects."""
        if isinstance(value, Lit):
            if value.datatype:
                return f'"{value.value}"^^<{value.datatype}>'
            else:
                return f'"{value.value}"'
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return f'"{str(value)}"'

    def _execute_update(self, query: str) -> None:
        """Execute a SPARQL UPDATE query."""
        if self.auth_type == "digest" and self.username and self.password:
            # Use requests for digest authentication
            headers = {"Content-Type": "application/sparql-update"}
            auth = HTTPDigestAuth(self.username, self.password)
            
            response = requests.post(
                self.endpoint,
                data=query,
                headers=headers,
                auth=auth
            )
            
            if response.status_code != 200:
                raise Exception(f"SPARQL update failed with status {response.status_code}: {response.text}")
        else:
            # Use SPARQLWrapper for basic auth or no auth
            self.sparql.setQuery(query)
            self.sparql.setMethod(POST)
            self.sparql.addCustomHttpHeader("Content-Type", "application/sparql-update")
            result = self.sparql.query()
            
            if result.getStatusCode() != 200:
                raise Exception(f"SPARQL update failed with status {result.getStatusCode()}: {result.getErrorMessage()}")

    def _execute_select(self, query: str) -> List[dict]:
        """Execute a SPARQL SELECT query and return results."""
        if self.auth_type == "digest" and self.username and self.password:
            # Use requests for digest authentication
            headers = {"Accept": "application/sparql-results+json"}
            auth = HTTPDigestAuth(self.username, self.password)
            
            response = requests.get(
                self.endpoint,
                params={"query": query},
                headers=headers,
                auth=auth
            )
            
            if response.status_code != 200:
                raise Exception(f"SPARQL select failed with status {response.status_code}: {response.text}")
            
            json_result = response.json()
            return json_result.get('results', {}).get('bindings', [])
        else:
            # Use SPARQLWrapper for basic auth or no auth
            self.sparql.setQuery(query)
            self.sparql.setMethod(GET)
            result = self.sparql.query()
            
            if result.getStatusCode() != 200:
                raise Exception(f"SPARQL select failed with status {result.getStatusCode()}: {result.getErrorMessage()}")
            
            json_result = result.convert(JSON)
            return json_result.get('results', {}).get('bindings', [])

    def create_entity(self, labels: List[str], props: Props | None = None, id: KGId = None) -> KGEntity:
        """Create an entity in the SPARQL endpoint."""
        if not labels:
            raise ValueError("At least one label is required")
        
        # Generate ID if not provided
        if id is None:
            from kgcore.common.types import new_id
            id = new_id()
        
        entity = KGEntity(id=id, labels=labels, props=props or {})
        if is_uri(id):
            entity_uri = id
        else:
            entity_uri = self._get_uri(id)
        
        # Build SPARQL INSERT query
        insert_parts = []
        
        # Add type assertions for labels
        for label in labels:
            label_uri = self._get_uri(label)
            insert_parts.append(f"<{entity_uri}> a <{label_uri}> .")
        
        # Add property assertions
        for prop, value in (props or {}).items():
            prop_uri = self._get_uri(prop)
            serialized_value = self._serialize_value(value)
            insert_parts.append(f"<{entity_uri}> <{prop_uri}> {serialized_value} .")
        
        query = f"INSERT DATA {{ GRAPH <{self.base_iri}> {{ {' '.join(insert_parts)} }} }}"
        
        try:
            self._execute_update(query)
        except Exception as e:
            raise Exception(f"Failed to create entity: {str(e)}")
        
        return entity

    def create_relation(self, type: str, source: KGId, target: KGId, props: Props | None = None) -> KGRelation:
        """Create a relation in the SPARQL endpoint."""
        relation = KGRelation(type=type, source=source, target=target, props=props or {})
        if is_uri(type):
            type_uri = type
        else:
            type_uri = self._get_uri(type)
        if is_uri(source):
            source_uri = source
        else:
            source_uri = self._get_uri(source)
        if is_uri(target):
            target_uri = target
        else:
            target_uri = self._get_uri(target)
        
        # Build SPARQL INSERT query
        insert_parts = []
        
        # Add the main triple
        insert_parts.append(f"<{source_uri}> <{type_uri}> <{target_uri}> .")
        
        # Add property assertions (using reification)
        if props:
            # Create a blank node for reification
            relation_uri = self._get_uri(relation.id)
            insert_parts.append(f"<{relation_uri}> a <{self._get_uri('Relation')}> .")
            insert_parts.append(f"<{relation_uri}> <{self._get_uri('subject')}> <{source_uri}> .")
            insert_parts.append(f"<{relation_uri}> <{self._get_uri('predicate')}> <{type_uri}> .")
            insert_parts.append(f"<{relation_uri}> <{self._get_uri('object')}> <{target_uri}> .")
            
            for prop, value in props.items():
                prop_uri = self._get_uri(prop)
                serialized_value = self._serialize_value(value)
                insert_parts.append(f"<{relation_uri}> <{prop_uri}> {serialized_value} .")
        
        query = f"INSERT DATA {{ GRAPH <{self.base_iri}> {{ {' '.join(insert_parts)} }} }}"
        
        try:
            self._execute_update(query)
        except Exception as e:
            raise Exception(f"Failed to create relation: {str(e)}")
        
        return relation

    def add_meta(self, target: KGId | tuple[KGId, str, KGId], meta: Props) -> None:
        """Add metadata to an entity, relation, or explicit triple using RDF reification."""
        if not meta:
            return
        
        # Handle different target types
        if isinstance(target, tuple):
            # Target is an explicit triple (subject, predicate, object)
            subject, predicate, object = target
            subject_uri = self._get_uri(subject)
            predicate_uri = self._get_uri(predicate)
            object_uri = self._get_uri(object)
            
            # Create reification for the explicit triple
            reification_id = f"reif_{subject}_{predicate}_{object}"
            reification_uri = self._get_uri(reification_id)
            
            insert_parts = [
                f"<{reification_uri}> a <{self._get_uri('Statement')}> .",
                f"<{reification_uri}> <{self._get_uri('subject')}> <{subject_uri}> .",
                f"<{reification_uri}> <{self._get_uri('predicate')}> <{predicate_uri}> .",
                f"<{reification_uri}> <{self._get_uri('object')}> <{object_uri}> ."
            ]
        else:
            # Target is an entity or relation ID
            target_uri = self._get_uri(target)
            
            # Find existing triples involving this target
            query = f"""
                SELECT ?s ?p ?o WHERE {{
                    {{ <{target_uri}> ?p ?o . BIND(<{target_uri}> as ?s) }}
                    UNION
                    {{ ?s ?p <{target_uri}> . BIND(<{target_uri}> as ?o) }}
                    UNION
                    {{ ?s <{target_uri}> ?o . BIND(<{target_uri}> as ?p) }}
                }}
            """
            
            try:
                results = self._execute_select(query)
                if not results:
                    raise ValueError(f"No triples found for target {target}")
                
                # Create reification for each found triple
                insert_parts = []
                for i, result in enumerate(results):
                    reification_id = f"reif_{target}_{i}"
                    reification_uri = self._get_uri(reification_id)
                    
                    s = result.get('s', {}).get('value', '')
                    p = result.get('p', {}).get('value', '')
                    o = result.get('o', {}).get('value', '')
                    
                    insert_parts.extend([
                        f"<{reification_uri}> a <{self._get_uri('Statement')}> .",
                        f"<{reification_uri}> <{self._get_uri('subject')}> <{s}> .",
                        f"<{reification_uri}> <{self._get_uri('predicate')}> <{p}> .",
                        f"<{reification_uri}> <{self._get_uri('object')}> <{o}> ."
                    ])
            except Exception as e:
                raise Exception(f"Failed to find triples for target {target}: {str(e)}")
        
        # Add metadata properties to reification nodes
        for prop, value in meta.items():
            prop_uri = self._get_uri(prop)
            serialized_value = self._serialize_value(value)
            # Add metadata to all reification nodes
            for part in insert_parts:
                if part.endswith(' .') and 'Statement' in part:
                    reification_uri = part.split()[0]
                    insert_parts.append(f"{reification_uri} <{prop_uri}> {serialized_value} .")
        
        query = f"INSERT DATA {{ GRAPH <{self.base_iri}default> {{ {' '.join(insert_parts)} }} }}"
        
        try:
            self._execute_update(query)
        except Exception as e:
            raise Exception(f"Failed to add metadata: {str(e)}")

    def find_entities(self, label: str | None = None, props: Props | None = None) -> List[KGEntity]:
        """Find entities matching the given criteria."""
        where_conditions = []
        
        if label:
            label_uri = self._get_uri(label)
            where_conditions.append(f"?entity a <{label_uri}> .")
        else:
            where_conditions.append("?entity a ?type .")
        
        # Add property conditions
        if props:
            for prop, value in props.items():
                prop_uri = self._get_uri(prop)
                serialized_value = self._serialize_value(value)
                where_conditions.append(f"?entity <{prop_uri}> {serialized_value} .")
        
        query = f"""
            SELECT DISTINCT ?entity ?type ?prop ?value WHERE {{
                {' '.join(where_conditions)}
                OPTIONAL {{ ?entity ?prop ?value . }}
            }}
        """
        
        try:
            results = self._execute_select(query)
            
            # Group results by entity
            entities = {}
            for result in results:
                entity_uri = result.get('entity', {}).get('value', '')
                if not entity_uri:
                    continue
                
                # Extract entity ID from URI
                entity_id = entity_uri.replace(self.base_iri, '')
                if entity_id not in entities:
                    entities[entity_id] = {
                        'id': entity_id,
                        'labels': [],
                        'props': {}
                    }
                
                # Add type/label
                if 'type' in result and result['type'].get('value'):
                    type_uri = result['type']['value']
                    type_id = type_uri.replace(self.base_iri, '')
                    if type_id not in entities[entity_id]['labels']:
                        entities[entity_id]['labels'].append(type_id)
                
                # Add property
                if 'prop' in result and 'value' in result and result['prop'].get('value') and result['value'].get('value'):
                    prop_uri = result['prop']['value']
                    prop_id = prop_uri.replace(self.base_iri, '')
                    value = result['value']['value']
                    entities[entity_id]['props'][prop_id] = value
            
            # Convert to KGEntity objects
            return [KGEntity(id=eid, labels=data['labels'], props=data['props']) 
                   for eid, data in entities.items()]
        
        except Exception as e:
            raise Exception(f"Failed to find entities: {str(e)}")

    def find_relations(self, type: str | None = None, props: Props | None = None) -> List[KGRelation]:
        """Find relations matching the given criteria."""
        where_conditions = []
        
        if type:
            type_uri = self._get_uri(type)
            where_conditions.append(f"?subject <{type_uri}> ?object .")
        else:
            where_conditions.append("?subject ?predicate ?object .")
        
        # Add property conditions (using reification)
        if props:
            where_conditions.extend([
                "?reif a <{self._get_uri('Statement')}> .",
                "?reif <{self._get_uri('subject')}> ?subject .",
                "?reif <{self._get_uri('predicate')}> ?predicate .",
                "?reif <{self._get_uri('object')}> ?object ."
            ])
            
            for prop, value in props.items():
                prop_uri = self._get_uri(prop)
                serialized_value = self._serialize_value(value)
                where_conditions.append(f"?reif <{prop_uri}> {serialized_value} .")
        
        query = f"""
            SELECT DISTINCT ?subject ?predicate ?object ?reif ?prop ?value WHERE {{
                {' '.join(where_conditions)}
                OPTIONAL {{ ?reif ?prop ?value . }}
            }}
        """
        
        try:
            results = self._execute_select(query)
            
            # Group results by relation
            relations = {}
            for result in results:
                subject_uri = result.get('subject', {}).get('value', '')
                predicate_uri = result.get('predicate', {}).get('value', '')
                object_uri = result.get('object', {}).get('value', '')
                
                if not all([subject_uri, predicate_uri, object_uri]):
                    continue
                
                # Create relation key
                subject_id = subject_uri.replace(self.base_iri, '')
                predicate_id = predicate_uri.replace(self.base_iri, '')
                object_id = object_uri.replace(self.base_iri, '')
                relation_key = f"{subject_id}_{predicate_id}_{object_id}"
                
                if relation_key not in relations:
                    relations[relation_key] = {
                        'type': predicate_id,
                        'source': subject_id,
                        'target': object_id,
                        'props': {}
                    }
                
                # Add property
                if 'prop' in result and 'value' in result and result['prop'].get('value') and result['value'].get('value'):
                    prop_uri = result['prop']['value']
                    prop_id = prop_uri.replace(self.base_iri, '')
                    value = result['value']['value']
                    relations[relation_key]['props'][prop_id] = value
            
            # Convert to KGRelation objects
            return [KGRelation(type=data['type'], source=data['source'], target=data['target'], props=data['props']) 
                   for data in relations.values()]
        
        except Exception as e:
            raise Exception(f"Failed to find relations: {str(e)}")


    def add_object(
        self,
        obj: Any,
        *,
        subject_ns: str = "urn:res:",
        predicate_ns: str = "urn:prop:",
        class_ns: str = "urn:cls:",
        type_predicate: URIRef = RDF.type,
    ) -> Graph:
        def to_predicate_uri(name: str) -> URIRef:
            if isinstance(name, str) and (name.startswith("http://") or name.startswith("https://") or name.startswith("urn:")):
                return URIRef(name)
            return URIRef(predicate_ns + str(name))

        def new_subject() -> URIRef:
            return URIRef(subject_ns + str(new_id()))

        # --- CHANGE 1: dump with aliases so Field(alias='@type') is preserved ---
        def to_mapping(value: Any) -> Mapping[str, Any] | None:
            if isinstance(value, BaseModel):
                # by_alias=True lets a field alias like '@type' show up
                return value.model_dump(by_alias=True, exclude_none=True)
            if isinstance(value, Mapping):
                return value
            return None

        def is_iterable(value: Any) -> bool:
            return isinstance(value, (list, tuple, set))

        # Helper: absolute IRI?
        def _is_abs_iri(s: Any) -> bool:
            return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://") or s.startswith("urn:"))

        # Helper: expand a possibly-curied name using @context or fallback class_ns
        # def _expand_type(typ: str, mapping: Mapping[str, Any] | None) -> URIRef:
        #     if _is_abs_iri(typ):
        #         return URIRef(typ)
        #     # Try JSON-LD @context CURIE expansion: {"@context": {"schema": "https://schema.org/"}}
        #     if mapping and isinstance(mapping.get("@context"), Mapping) and ":" in typ:
        #         prefix, local = typ.split(":", 1)
        #         base = mapping["@context"].get(prefix)
        #         if isinstance(base, str) and _is_abs_iri(base):
        #             return URIRef(base.rstrip("#/") + "/" + local)
        #     # Fallback: prefix with class_ns
        #     return URIRef(class_ns + typ)

        # # --- CHANGE 2: more tolerant dict type detection, with lists and CURIEs ---
        # def _dict_type_iri(mapping: Mapping[str, Any] | None) -> URIRef | None:
        #     if not mapping:
        #         return None
        #     # Check several common keys
        #     candidates = mapping.get("@type") or mapping.get("type") or mapping.get("rdf_type") or mapping.get("__rdf_type__")
        #     if candidates is None:
        #         return None
        #     # Allow list of types; pick the first by default (or adapt to your policy)
        #     if is_iterable(candidates):
        #         for c in candidates:
        #             if isinstance(c, str):
        #                 return _expand_type(c, mapping)
        #         return None
        #     if isinstance(candidates, str):
        #         return _expand_type(candidates, mapping)
        #     return None

        def class_iri_for(value: Any, mapping: Mapping[str, Any] | None) -> URIRef | None:

            # 1) instance or class-level __rdf_type__
            iri = getattr(value, "__rdf_type__", None) or getattr(type(value), "__rdf_type__", None)
            if _is_abs_iri(iri):
                return URIRef(iri)

            # 2) model_config on instance or class
            cfg = getattr(value, "model_config", None) or getattr(type(value), "model_config", None) or {}
            iri = (cfg or {}).get("rdf_type")
            if _is_abs_iri(iri):
                return URIRef(iri)

            # 3) enhanced dict-type detection (works for dumped models if you aliased a field)
            if mapping:
                iri = mapping.get("@type") or mapping.get("type") or mapping.get("rdf_type") or mapping.get("__rdf_type__")
                if iri:
                    return URIRef(iri)

            # 4) fallback for BaseModel -> urn:cls:ClassName
            if isinstance(value, BaseModel):
                return URIRef(class_ns + value.__class__.__name__)

            return None

        g = Graph()
        root_map = to_mapping(obj)
        if root_map is None:
            raise TypeError("add_object expects a Pydantic BaseModel or a dict-like object.")

        root_subject = new_subject()

        root_type = class_iri_for(obj, root_map)
        if root_type is not None:
            g.add((root_subject, type_predicate, root_type))

        def add_value(head: URIRef, key: str, value: Any) -> None:
            if value is None:
                return

            nested_map = to_mapping(value)
            if nested_map is not None:
                tail = new_subject()
                g.add((head, to_predicate_uri(key), tail))
                t = class_iri_for(value, nested_map)
                if t is not None:
                    g.add((tail, type_predicate, t))
                for k, v in nested_map.items():
                    if k in ("@type", "type", "rdf_type"):  # already emitted
                        continue
                    add_value(tail, k, v)
                return

            if is_iterable(value):
                for item in value:
                    add_value(head, key, item)
                return

            lit = Literal(value) if isinstance(value, (str, int, float, bool)) else Literal(str(value))
            g.add((head, to_predicate_uri(key), lit))

        for k, v in root_map.items():
            if k in ("@type", "type", "rdf_type"):
                continue
            add_value(root_subject, k, v)

        return g


    # def add_object(
    #     self,
    #     obj: Any,
    #     *,
    #     subject_ns: str = "urn:res:",
    #     predicate_ns: str = "urn:prop:",
    #     class_ns: str = "urn:cls:",
    #     # If your models carry an explicit RDF type, look here first
    #     # - attribute:  value.__rdf_type__  (e.g., "https://schema.org/Person")
    #     # - config:     value.model_config.get("rdf_type")
    #     # For dicts, you can also put "@type": "..." inside the dict.
    #     type_predicate: URIRef = RDF.type,  # usually rdf:type
    # ) -> Graph:
    #     def to_predicate_uri(name: str) -> URIRef:
    #         if isinstance(name, str) and (name.startswith("http://") or name.startswith("https://") or name.startswith("urn:")):
    #             return URIRef(name)
    #         return URIRef(predicate_ns + str(name))

    #     def new_subject() -> URIRef:
    #         return URIRef(subject_ns + str(new_id()))

    #     def to_mapping(value: Any) -> Mapping[str, Any] | None:
    #         if isinstance(value, BaseModel):
    #             return value.model_dump()
    #         if isinstance(value, Mapping):
    #             return value
    #         return None

    #     def is_iterable(value: Any) -> bool:
    #         return isinstance(value, (list, tuple, set))

    #     def class_iri_for(value: Any, mapping: Mapping[str, Any] | None) -> URIRef | None:
    #         """
    #         Compute the rdf:type IRI for a BaseModel or dict.
    #         Priority:
    #         1) explicit attr  value.__rdf_type__ (IRI)
    #         2) pydantic v2 config value.model_config.get("rdf_type")
    #         3) dict key '@type' (IRI)
    #         4) fallback to class_ns + ClassName for BaseModel
    #         """
    #         # 1) explicit attribute on the instance
    #         iri = getattr(value, "__rdf_type__", None)
    #         if isinstance(iri, str) and (iri.startswith("http://") or iri.startswith("https://") or iri.startswith("urn:")):
    #             return URIRef(iri)

    #         # 2) pydantic model_config hint
    #         if isinstance(value, BaseModel):
    #             cfg = getattr(value, "model_config", {}) or {}
    #             iri = cfg.get("rdf_type")
    #             if isinstance(iri, str) and (iri.startswith("http://") or iri.startswith("https://") or iri.startswith("urn:")):
    #                 return URIRef(iri)

    #         # 3) dict '@type'
    #         if mapping is not None:
    #             iri = mapping.get("@type") or mapping.get("__rdf_type__")
    #             if isinstance(iri, str) and (iri.startswith("http://") or iri.startswith("https://") or iri.startswith("urn:")):
    #                 return URIRef(iri)

    #         # 4) fallback for BaseModel -> urn:cls:ClassName
    #         if isinstance(value, BaseModel):
    #             return URIRef(class_ns + value.__class__.__name__)

    #         return None

    #     g = Graph()

    #     root_map = to_mapping(obj)
    #     if root_map is None:
    #         raise TypeError("add_object expects a Pydantic BaseModel or a dict-like object.")

    #     root_subject = new_subject()

    #     # --- NEW: add rdf:type for the root ---
    #     root_type = class_iri_for(obj, root_map)
    #     if root_type is not None:
    #         g.add((root_subject, type_predicate, root_type))

    #     def add_value(head: URIRef, key: str, value: Any) -> None:
    #         if value is None:
    #             return

    #         nested_map = to_mapping(value)
    #         if nested_map is not None:
    #             tail = new_subject()
    #             # link parent to child
    #             g.add((head, to_predicate_uri(key), tail))

    #             # --- NEW: add rdf:type for the nested object ---
    #             t = class_iri_for(value, nested_map)
    #             if t is not None:
    #                 g.add((tail, type_predicate, t))

    #             for k, v in nested_map.items():
    #                 # skip @type key since we already emitted rdf:type
    #                 if k == "@type":
    #                     continue
    #                 add_value(tail, k, v)
    #             return

    #         if is_iterable(value):
    #             for item in value:
    #                 add_value(head, key, item)
    #             return

    #         lit = Literal(value) if isinstance(value, (str, int, float, bool)) else Literal(str(value))
    #         g.add((head, to_predicate_uri(key), lit))

    #     for k, v in root_map.items():
    #         if k == "@type":
    #             continue  # already handled
    #         add_value(root_subject, k, v)

    #     return g

    # def add_object(
    #     self,
    #     obj: Any,
    #     *,
    #     subject_ns: str = "urn:res:",     # for generated subject URIs
    #     predicate_ns: str = "urn:prop:"   # for non-IRI keys
    # ) -> Graph:
    #     """
    #     Create an RDF graph from a Pydantic model (or plain dict-like).
    #     - Nested dicts/lists are handled recursively.
    #     - Each embedded object becomes a new subject URI linked from its parent.
    #     - Predicates are turned into URIRefs. If a key isn't an absolute IRI,
    #       it's prefixed with `predicate_ns`.
    #     - Primitive values become Literals.
    #     Returns the constructed rdflib.Graph. The root subject is a fresh URI.
    #     """

    #     def to_predicate_uri(name: str) -> URIRef:
    #         if isinstance(name, str) and (
    #             name.startswith("http://")
    #             or name.startswith("https://")
    #             or name.startswith("urn:")
    #         ):
    #             return URIRef(name)
    #         return URIRef(predicate_ns + str(name))

    #     def new_subject() -> URIRef:
    #         # new_id() is assumed to return a unique string identifier
    #         return URIRef(subject_ns + str(new_id()))

    #     def to_mapping(value: Any) -> Mapping[str, Any] | None:
    #         if isinstance(value, BaseModel):
    #             return value.model_dump()
    #         if isinstance(value, Mapping):
    #             return value
    #         return None

    #     def is_iterable(value: Any) -> bool:
    #         return isinstance(value, (list, tuple, set))

    #     graph = Graph()

    #     # Normalize the root value into a mapping
    #     root_map = to_mapping(obj)
    #     if root_map is None:
    #         raise TypeError("add_object expects a Pydantic BaseModel or a dict-like object.")

    #     root_subject = new_subject()

    #     def add_value(head: URIRef, key: str, value: Any) -> None:
    #         # Skip Nones
    #         if value is None:
    #             return

    #         # Allow nested BaseModels
    #         nested_map = to_mapping(value)
    #         if nested_map is not None:
    #             tail = new_subject()
    #             graph.add((head, to_predicate_uri(key), tail))
    #             for k, v in nested_map.items():
    #                 add_value(tail, k, v)
    #             return

    #         # Recurse lists/tuples/sets (re-using same predicate)
    #         if is_iterable(value):
    #             for item in value:
    #                 add_value(head, key, item)
    #             return

    #         # Otherwise, add as a literal (stringify unknowns)
    #         if isinstance(value, (str, int, float, bool)):
    #             lit = Literal(value)
    #         else:
    #             lit = Literal(str(value))

    #         graph.add((head, to_predicate_uri(key), lit))

    #     # Populate graph from root
    #     for k, v in root_map.items():
    #         add_value(root_subject, k, v)

    #     # Optionally, expose the root subject IRI for the caller if needed:
    #     # graph.bind("res", subject_ns)  # you can bind prefixes if you like
    #     # graph.bind("prop", predicate_ns)

    #     return graph

    # def add_object(self,obj: Any):
        # if not isinstance(obj, BaseModel):
        #     print("not seriaziable")
        #     pass
        # root_jd = obj.model_dump()

        # from rdflib import Graph
        # from kgcore.common.types import new_id
        # graph = Graph()


        # def add_json_object(jd, head: str = "", rel: str = ""):
        #     if head == "":
        #         tail = new_id
            
        #     for k,v in jd.items():
        #         if isinstance(v, List):
        #             for _v in v:
        #                 add_json_object(_v,k)
        #             return
        #         if isinstance(v,Dict):
        #             for _k,_v in v.items():
        #                 add_json_object(_v,_k)
        #             return
        #         if rel == "":
        #             graph.add(URIRef(new_id),URIRef(k),Literal(v))
        #         else:
        #             graph.add(URIRef())
                




        