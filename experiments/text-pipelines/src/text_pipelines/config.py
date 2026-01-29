from enum import Enum
from kgpipe.common.model.data import DynamicFormat, FormatRegistry

class ExtendedFormats(Enum):
    SPECIAL_IN = DynamicFormat(name="special_in", extension=".special_in", description="Special input format")
    SPECIAL1 = DynamicFormat(name="special1", extension=".special1", description="Special format 1")
    SPECIAL2 = DynamicFormat(name="special2", extension=".special2", description="Special format 2")
    SPECIAL_KG = DynamicFormat(name="special_kg", extension=".special_kg", description="Special output format for knowledge graph")

FORMAT_REGISTRY = FormatRegistry()

FORMAT_REGISTRY.register_format(
    ExtendedFormats.SPECIAL_IN.value.name, ExtendedFormats.SPECIAL_IN.value.extension, ExtendedFormats.SPECIAL_IN.value.description)
FORMAT_REGISTRY.register_format(
    ExtendedFormats.SPECIAL1.value.name, ExtendedFormats.SPECIAL1.value.extension, ExtendedFormats.SPECIAL1.value.description)
FORMAT_REGISTRY.register_format(
    ExtendedFormats.SPECIAL2.value.name, ExtendedFormats.SPECIAL2.value.extension, ExtendedFormats.SPECIAL2.value.description)
FORMAT_REGISTRY.register_format(
    ExtendedFormats.SPECIAL_KG.value.name, ExtendedFormats.SPECIAL_KG.value.extension, ExtendedFormats.SPECIAL_KG.value.description)