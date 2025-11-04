# reimport of tasks from kgpipe

from kgpipe_tasks.entity_resolution import(
    paris_entity_matching,
    paris_exchange,
    pyjedai_entity_matching,
    pyjedai_entity_matching_v2
)

from kgpipe_tasks.entity_resolution.fusion import (
    fusion_first_value,
    select_first_value,
    aggregate_2matches,
    reduce_to_best_match_per_entity
)

from kgpipe_tasks.construction import (
    construct_rdf_from_json,
    construct_rdf_from_te_json,
    construct_linkedrdf_from_json,
    construct_te_document_from_json
)
from kgpipe_tasks.text_processing import (
    corenlp_openie_extraction,
    corenlp_exchange,
    dbpedia_spotlight_ner_nel,
    dbpedia_spotlight_exchange,
)
from kgpipe_tasks.transform_interop import (
    aggregate2_te_json,
    aggregate3_te_json,
    transform2_rdf_to_csv_v2
)

__all__ = [
    "paris_entity_matching",
    "paris_exchange",
    "jedai_tab_matcher",
    "construct_rdf_from_json",
    "construct_rdf_from_te_json",
    "construct_rdf_from_te_json_mappings_only",
    "construct_te_document_from_json",
    "corenlp_openie_extraction",
    "corenlp_exchange",
    "dbpedia_spotlight_ner_nel",
    "dbpedia_spotlight_exchange",
    "aggregate2_te_json",
    "aggregate3_te_json",
    "fusion_union_rdf",
    "aggregate_2matches",
    "union_matched_rdf",
    "union_matched_rdf_combined",
    "reduce_to_best_match_per_entity",
    "valentine_csv_matching_v2",
    "valentine_csv_matching"
    "transform2_rdf_to_csv_v2"
]