from rdflib import OWL, RDFS
from kgpipe.common.systemgraph import SYS_KG
from kgcore.api import KGProperty
from typing import get_origin, get_args, Union

# TODO move to kgcore.api 
# TODO handle resolution of SYS_KG
def kg_class(description: str = ""):
    """
    Decorator factory: creates a decorator that registers the class
    as a KG entity (type/Class node) once at import time.
    """
    def decorator(cls):
        print("kg_class decorator called for class: ", cls.__name__)
        # add owl class
        props = []
        if description:
            props.append(KGProperty(key="description", value=description))
        class_et = SYS_KG.create_entity(id=cls.__name__, types=[OWL.Class], properties=props)
        # get variables of the class
        # for each variable, add a DatatypeProperty or ObjectProperty
        for var in cls.__annotations__.keys():
            annotation = cls.__annotations__[var]
            is_object_property = False
            with_object_type = None
            
            # Check if it's a direct class type (e.g., Data)
            if isinstance(annotation, type) and annotation not in (str, int, float, bool):
                is_object_property = True
                with_object_type = annotation.__name__
            else:
                # Check for generic types (List, Dict, Optional, etc.)
                origin = get_origin(annotation)
                args = get_args(annotation)
                
                # Check if any of the type arguments are classes (not primitives)
                # Note: List[str], List[int], etc. remain DatatypeProperty because
                # str, int, float, bool are primitive types, not class types
                for arg in args:
                    if isinstance(arg, type):
                        # Skip primitive types - these result in DatatypeProperty
                        # Only non-primitive class types result in ObjectProperty
                        if arg not in (str, int, float, bool, type(None)):
                            is_object_property = True
                            with_object_type = arg.__name__
                            break
                    # Handle string forward references (e.g., "Data" in quotes)
                    elif isinstance(arg, str):
                        # Try to resolve the string reference
                        # For now, assume string references might be classes
                        is_object_property = True
                        break
            
            # Also check if it's a class attribute (old behavior for dataclasses)
            if not is_object_property:
                attr_value = getattr(cls, var, None)
                if attr_value is not None and isinstance(attr_value, type) and attr_value not in (str, int, float, bool):
                    is_object_property = True
            
            if is_object_property:
                # add ObjectProperty
                prop_et = SYS_KG.create_entity(id=cls.__name__+"_"+var, types=[OWL.ObjectProperty], properties={
                    RDFS.label: var,
                })
                SYS_KG.create_relation(source=prop_et.id, target=class_et.id, type=str(RDFS.domain))
                if with_object_type:
                    SYS_KG.create_relation(source=prop_et.id, target=with_object_type, type=str(RDFS.range))
            else:
                # add DatatypeProperties
                prop_et = SYS_KG.create_entity(id=cls.__name__+"_"+var, types=[OWL.DatatypeProperty], properties={
                    RDFS.label: var,
                })
                SYS_KG.create_relation(source=prop_et.id, target=class_et.id, type=str(RDFS.domain))
        return cls

    return decorator