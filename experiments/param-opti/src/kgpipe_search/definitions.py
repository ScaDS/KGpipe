from pydantic import BaseModel
from typing import List, Dict, Optional
from kgpipe.common import KgTask
from kgpipe.common.model.configuration import ConfigurationProfile
from pathlib import Path

class PipelineLayout(BaseModel):
    """
    allowed task categories in the pipeline
    """
    allowed_task_categories: List[str]


class PipelineConfig(BaseModel):
    tasks: List[KgTask]
    config_catalog: Dict[str, ConfigurationProfile]
    result_path: Optional[Path] = None
    seed_path: Optional[Path] = None



RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "rdf_sampled_pipeline_configs.json"
_RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1

RDF_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "rdf_unique_sampled_pipeline_configs.json"
_RDF_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1

RDF_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "rdf_exhaustive_pipeline_configs.json"
_RDF_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1

TEXT_SAMPLED_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "text_sampled_pipeline_configs.json"
_TEXT_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1

TEXT_UNIQUE_SAMPLED_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "text_unique_sampled_pipeline_configs.json"
_TEXT_UNIQUE_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1

TEXT_EXHAUSTIVE_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "text_exhaustive_pipeline_configs.json"
_TEXT_EXHAUSTIVE_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1

TEXT_PIPELINE_LAYOUT = PipelineLayout(
    allowed_task_categories=["information_extraction", "entity_linking", "aggregate_entity_linking", "relation_linking", "aggregate_relation_linking", "construct_rdf", "fusion"]
) 

RDF_PIPELINE_LAYOUT = PipelineLayout(
    allowed_task_categories=["ontology_matching", "entity_matching", "aggregate_matching_results", "fusion"]
)

RDF_LAYOUT = [
    "graph_alignment"
    "relation_matching"
    "entity_matching"
    "aggregate_matching"
]


RDF_SEARCH_SPACE = {
    "graph_alignment_label_alias_embedding_transformer_task": {
        "category": ["ontology_matching", "entity_matching", "aggregate_matching_results"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "relation_matcher_label_alias_embedding_transformer_task": {
        "category": ["ontology_matching"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_matcher_label_alias_embedding_transformer_task": {
        "category": ["entity_matching"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "paris_ontology_matching_task": {
        "category": ["ontology_matching"],
        "ontology_matching_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "paris_entity_alignment_task": {
        "category": ["entity_matching"],
        "entity_matching_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "paris_graph_alignment_task": {
        "category": ["ontology_matching", "entity_matching", "aggregate_matching_results"],
        "entity_matching_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
        "relation_matching_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "aggregate_matching_results_task": {
        "category": ["aggregate_matching_results"],
    },
    "fusion_first_value_task": {
        "category": ["fusion"],
        # "fusion_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "relation_linker_label_alias_embedding_transformer_task": {
        "category": ["entity_linking"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_linker_label_alias_embedding_transformer_task": {
        "category": "entity_linking",
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
}

RDF_BASELINE_CONFIG =  {
      "profiles": {
        "paris_graph_alignment_task": {
          "bindings": [
            {
              "parameter": "entity_matching_threshold",
              "value": 0.9
            },
            {
              "parameter": "relation_matching_threshold",
              "value": 0.5
            }
          ],
          "profile_name": "paris_graph_alignment_entity_matching_threshold=0.9,relation_matching_threshold=0.5"
        }
      },
      "task_keys": [
        "paris_graph_alignment_task",
        "fusion_first_value_task"
      ]
    }

TEXT_LAYOUT = [
    "information_extraction"
    "entity_linking"
    "relation_linking"
    "fusion"
]

TEXT_SEARCH_SPACE = {
    "corenlp_text_extraction_task": {
        "category": ["information_extraction"],
        # does not have config parameters
    },
    "genie_text_extraction_task": {
        "category": ["information_extraction"],
        # does not have config parameters
    },
    "spotlight_entity_linking_task": {
        "category": ["entity_linking"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "relation_linker_label_alias_embedding_transformer_task": {
        "category": ["relation_linking"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_linker_label_alias_embedding_transformer_task": {
        "category": ["entity_linking"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "aggregate_entity_linking_task": {
        "category": ["aggregate_entity_linking"],
    },
    "aggregate_relation_linking_task": {
        "category": ["aggregate_relation_linking"],
    },
    "generate_rdf_from_text_results_task": {
        "category": ["construct_rdf"],
    },
    "select_first_value_task": {
        "category": ["fusion"],
    },
}

from kgpipe_search.dev.tasks.paris import paris_graph_alignment_task, paris_entity_alignment_task, paris_ontology_matching_task
from kgpipe_search.dev.tasks.fusion import fusion_first_value_task
from kgpipe_search.dev.tasks.base_linker import relation_linker_label_alias_embedding_transformer_task, entity_linker_label_alias_embedding_transformer_task
from kgpipe_search.dev.tasks.base_matcher import (
    graph_alignment_label_alias_embedding_transformer_task,
    relation_matcher_label_alias_embedding_transformer_task,
    entity_matcher_label_alias_embedding_transformer_task,
)
from kgpipe_search.dev.tasks.corenlp import corenlp_text_extraction_task
from kgpipe_search.dev.tasks.genie import genie_text_extraction_task
from kgpipe_search.dev.tasks.spotlight import spotlight_entity_linking_task
from kgpipe_search.dev.tasks.matching_helpers import aggregate_matching_results_task
from kgpipe_search.dev.tasks.text_helpers import aggregate_entity_linking_task, aggregate_relation_linking_task
from kgpipe_search.dev.tasks.text_helpers import generate_rdf_from_text_results_task
from kgpipe_search.dev.tasks.select_lib import select_first_value_task

TEXT_TASK_DICT = {
    "corenlp_text_extraction_task": corenlp_text_extraction_task,
    "genie_text_extraction_task": genie_text_extraction_task,
    "spotlight_entity_linking_task": spotlight_entity_linking_task,
    "relation_linker_label_alias_embedding_transformer_task": relation_linker_label_alias_embedding_transformer_task,
    "entity_linker_label_alias_embedding_transformer_task": entity_linker_label_alias_embedding_transformer_task,
    "select_first_value_task": select_first_value_task,
    "aggregate_entity_linking_task": aggregate_entity_linking_task,
    "aggregate_relation_linking_task": aggregate_relation_linking_task,
    "generate_rdf_from_text_results_task": generate_rdf_from_text_results_task,
}

RDF_TASK_DICT = {
    "graph_alignment_label_alias_embedding_transformer_task": graph_alignment_label_alias_embedding_transformer_task,
    "relation_matcher_label_alias_embedding_transformer_task": relation_matcher_label_alias_embedding_transformer_task,
    "entity_matcher_label_alias_embedding_transformer_task": entity_matcher_label_alias_embedding_transformer_task,
    "paris_ontology_matching_task": paris_ontology_matching_task,
    "paris_entity_alignment_task": paris_entity_alignment_task,
    "paris_graph_alignment_task": paris_graph_alignment_task,
    "fusion_first_value_task": fusion_first_value_task,
    "relation_linker_label_alias_embedding_transformer_task": relation_linker_label_alias_embedding_transformer_task,
    "entity_linker_label_alias_embedding_transformer_task": entity_linker_label_alias_embedding_transformer_task,
    "aggregate_matching_results_task": aggregate_matching_results_task,
    # "fusion_union_task": fusion_union_task,
}

task_dict = {**TEXT_TASK_DICT, **RDF_TASK_DICT}