
HEADERS = ["pipeline", "stage", "aspect", "metric", "value", "normalized", "duration", "details"]

# Only keep these classes and aggregate the rest into "Other"
main_classes = [
    "http://kg.org/ontology/Company",
    "http://kg.org/ontology/Person",
    "http://kg.org/ontology/Film"
]

name_mapping = {
    "rdf_a": r"\sspRDFa",
    "rdf_b": r"\sspRDFb",
    "rdf_c": r"\sspRDFc",
    "rdf_llm_schema_align_v1": r"\sspRDFc",
    "json_a": r"\sspJSONa",
    "json_b": r"\sspJSONb",
    "json_c": r"\sspJSONc",
    "json_llm_mapping_v1": r"\sspJSONc",
    "text_a": r"\sspTexta",
    "text_b": r"\sspTextb",
    "text_c": r"\sspTextc",
    "text_llm_triple_extract_v1": r"\sspTextc",
    "rdf_json_text": r"\mspRJT",
    "rdf_text_json": r"\mspRTJ",
    "json_rdf_text": r"\mspJRT",
    "json_text_rdf": r"\mspJTR",
    "text_rdf_json": r"\mspTRJ",
    "text_json_rdf": r"\mspTJR",
}

METRIC_NAME_MAP = {
    "entity_count": "EC",
    "relation_count": "RC",
    "triple_count": "FC",
    "class_count": "TC",
    "duration": "Time",
    "loose_entity_count": "LEC",
    "shallow_entity_count": "SEC",
    # Semantic/Reasoning metrics
    "reasoning": "EO",
    "disjoint_domain": "EO1",
    "incorrect_relation_direction": "EO2",
    "incorrect_relation_cardinality": "EO3",
    "incorrect_relation_range": "EO4",
    "incorrect_relation_domain": "EO5",
    "incorrect_datatype": "EO6",
    "incorrect_datatype_format": "EO7",
    "ontology_class_coverage": "EO8",
    "ontology_relation_coverage": "EO9",
    "ontology_namespace_coverage": "E10",
    # Reference metrics
    "ReferenceTripleAlignmentMetric": "RTC",
    "ReferenceTripleAlignmentMetricSoftE": "RTC-SoftE",
    "ReferenceTripleAlignmentMetricSoftEV": "RTC-SoftEV",
    "ReferenceClassCoverageMetric": "RCC",
    # ER metrics
    "ER_EntityMatchMetric": "ER-EM",
    "ER_RelationMatchMetric": "ER-RM",
    # TE metrics
    "TE_ExpectedEntityLinkMetric": "TE-EEL",
    "TE_ExpectedRelationLinkMetric": "TE-ERL",
    # Source metrics
    "SourceEntityCoverageMetric": "VSEC",
    "SourceEntityCoverageMetricSoft": "VSEC-Soft",
    "REI_precision": "REI-Precision",

}

# long: 
# disjoint_domain
# incorrect_relation_domain
# incorrect_relation_range
# incorrect_relation_direction
# incorrect_datatype
# incorrect_datatype_format
# short:ODT OD OR ORD OLT OLF OAvg
SEM_METRIC_SHORT_NAMES = {
    # "reasoning" : "EO0",
    "disjoint_domain": "$O_{DT}$",
    "incorrect_relation_direction": "$O_{RD}$",
    "incorrect_relation_cardinality": "$O_{CA}$",
    "incorrect_relation_range": "$O_{R}$",
    "incorrect_relation_domain": "$O_{D}$",
    "incorrect_datatype": "$O_{LT}$",
    "incorrect_datatype_format": "$O_{LF}$",
    # "ontology_class_coverage": "$O_{CC}$",
    # "ontology_relation_coverage": "$O_{RC}$",
    # "ontology_namespace_coverage": "$O_{NC}$",
}

METRIC_NAME_INDEX_PRETTY = [
    ("duration", "Runtime Duration"),
    ("triple_count", "Fact/Triple Count"),
    ("entity_count", "Entity Count"),
    ("relation_count", "Relation Count"),
    ("class_count", "Entity Type Count"),
    ("Person", "Persons"),
    ("Film", "Films"),
    ("Company", "Companies"),
    ("Other", "Other Type"),
    ("loose_entity_count", "Empty Entities"),
    ("shallow_entity_count", "Shallow Entities"),
    # Semantic/Reasoning metrics
    # ("reasoning", "Reasoning"),
    ("disjoint_domain", "Disjoint Domain"),
    ("incorrect_relation_direction", "Incorrect Relation Direction"),
    ("incorrect_relation_cardinality", "Incorrect Relation Cardinality"),
    ("incorrect_relation_range", "Incorrect Relation Range"),
    ("incorrect_relation_domain", "Incorrect Relation Domain"),
    ("incorrect_datatype", "Incorrect Datatype"),
    ("incorrect_datatype_format", "Incorrect Datatype Format"),
    ("ontology_class_coverage", "Ontology Class Coverage"),
    ("ontology_relation_coverage", "Ontology Relation Coverage"),
    ("ontology_namespace_coverage", "Ontology Namespace Coverage"),
    # Source metrics
    ("SourceEntityCoverageMetric", "Source Entity Recall"),
    ("SourceEntityCoverageMetricSoft", "Source Entity Recall (~ID)"),
    ("REI_precision", "Source Entity Precision (~ID)"),
    # Reference metrics
    ("ReferenceTripleAlignmentMetric", "Reference Alignment (f1)"),
    ("ReferenceTripleAlignmentMetricSoftE", "Reference Alignment (~ID) (f1)"),
    ("ReferenceTripleAlignmentMetricSoftEV", "Reference Alignment (~ID~Value) (f1)"),
    # ("ReferenceClassCoverageMetric", "Reference Class Coverage"),
    # ER metrics
    ("ER_EntityMatchMetric", "Entity Match (p)"),
    ("ER_RelationMatchMetric", "Relation Match (p)"),
    # TE metrics
    ("TE_ExpectedEntityLinkMetric", "Expected Entity Link (p)"),
    ("TE_ExpectedRelationLinkMetric", "Expected Relation Link (p)"),
]

METRIC_NAME_MAP_PRETTY = {k: v for k, v in METRIC_NAME_INDEX_PRETTY}
SEM_METRIC_LONG_NAMES = {v: k for k, v in SEM_METRIC_SHORT_NAMES.items()}
