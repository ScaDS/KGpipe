from kgpipe.common import KgPipe, Data, DataFormat
from kgpipe.common.model.configuration import ConfigurationProfile, ParameterBinding, ConfigurationDefinition
from param_opti.tasks.paris import paris_graph_alignment_task, paris_entity_alignment_task, paris_ontology_matching_task
from param_opti.tasks.fusion import fusion_first_value_task
from param_opti.tasks.base_linker import relation_linker_label_alias_embedding_transformer_task, entity_linker_label_alias_embedding_transformer_task
from param_opti.tasks.corenlp import corenlp_text_extraction_task
from param_opti.tasks.genie import genie_text_extraction_task
from param_opti.tasks.spotlight import spotlight_entity_linking_task
from param_opti.tasks.text_helpers import aggregate_text_tasks_task, generate_rdf_from_text_results_task
from param_opti.tasks.select_lib import select_first_value_task
from qap.test_pipeline_config import (
    PipelineConfig,
    _get_param,
    load_rdf_sampled_pipeline_configs,
)
from pathlib import Path
import pytest
import os

from dotenv import load_dotenv
load_dotenv()

tmp_base_dir = Path("tmp")
if not tmp_base_dir.exists():
    tmp_base_dir.mkdir(parents=True, exist_ok=True)


ontology_path = "data/input_final/target_kg/ontology.ttl"
os.environ["ONTOLOGY_PATH"] = ontology_path


def get_default_rdf_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        tasks=[
            paris_graph_alignment_task,
            fusion_first_value_task,
        ],
        config_catalog={
            # Key must match KgTask.name because KgPipe delegates by task.name
            "paris_graph_alignment": ConfigurationProfile(
                name="paris_graph_alignment",
                definition=paris_graph_alignment_task.config_spec,
                bindings=[
                    ParameterBinding(parameter=_get_param(paris_graph_alignment_task.config_spec, "entity_matching_threshold"), value=0.5),
                    ParameterBinding(parameter=_get_param(paris_graph_alignment_task.config_spec, "relation_matching_threshold"), value=0.5),
                ],
            )
        },
    )

def test_rdf_pipeline_from_default_config():
    pipeline_config = get_default_rdf_pipeline_config()

    seed_path = tmp_base_dir / "seed.nt"
    source_path = tmp_base_dir / "source.nt"
    result_path = tmp_base_dir / "result.nt"
    tasks_tmp_dir = tmp_base_dir / "tasks_tmp"
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Ensure inputs exist for pipeline execution.
    seed_path.write_text("<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n")
    source_path.write_text("<http://example.org/s2> <http://example.org/p> <http://example.org/o> .\n")

    pipeline = KgPipe(
        tasks=pipeline_config.tasks, 
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name="test_pipeline")

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES), 
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES))

    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=True)


@pytest.mark.parametrize("config_idx", range(len(load_rdf_sampled_pipeline_configs())))
def test_rdf_pipeline_from_saved_sampled_configs(config_idx):
    """Runs KGpipe using PipelineConfigs materialized from the JSON fixture written by test_pipeline_config."""
    configs = load_rdf_sampled_pipeline_configs()
    assert configs, "fixtures/rdf_sampled_pipeline_configs.json is missing or empty; run test_enumerate_all_valid_rdf_task_combinations_with_config_sampling"

    pipeline_config = configs[config_idx]

    seed_path = tmp_base_dir / "seed_saved_sample.nt"
    source_path = tmp_base_dir / "source_saved_sample.nt"
    result_path = tmp_base_dir / f"result_saved_sample_config_idx_{config_idx}.nt"
    tasks_tmp_dir = tmp_base_dir / f"tasks_tmp_saved_sample_config_idx_{config_idx}"
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

    seed_path.write_text("<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n")
    source_path.write_text("<http://example.org/s2> <http://example.org/p> <http://example.org/o> .\n")

    pipeline = KgPipe(
        tasks=pipeline_config.tasks,
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name="test_pipeline_saved_sample",
    )

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )

    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=True)



def get_default_text_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        tasks=[
            corenlp_text_extraction_task,
            entity_linker_label_alias_embedding_transformer_task,
            relation_linker_label_alias_embedding_transformer_task,
            aggregate_text_tasks_task,  
            generate_rdf_from_text_results_task,
            select_first_value_task,
        ],
        config_catalog={
            "entity_linker_label_alias_embedding_transformer": ConfigurationProfile(
                name="entity_linker_label_alias_embedding_transformer",
                definition=entity_linker_label_alias_embedding_transformer_task.config_spec,
                bindings=[
                    ParameterBinding(parameter=_get_param(entity_linker_label_alias_embedding_transformer_task.config_spec, "model_name"), value="sentence-transformers/all-MiniLM-L6-v2"),
                    ParameterBinding(parameter=_get_param(entity_linker_label_alias_embedding_transformer_task.config_spec, "similarity_threshold"), value=0.5),
                ],
            ),
            "relation_linker_label_alias_embedding_transformer": ConfigurationProfile(
                name="relation_linker_label_alias_embedding_transformer",
                definition=relation_linker_label_alias_embedding_transformer_task.config_spec,
                bindings=[
                    ParameterBinding(parameter=_get_param(relation_linker_label_alias_embedding_transformer_task.config_spec, "model_name"), value="sentence-transformers/all-MiniLM-L6-v2"),
                    ParameterBinding(parameter=_get_param(relation_linker_label_alias_embedding_transformer_task.config_spec, "similarity_threshold"), value=0.5),
                ],
            ),
        },
    )

def test_text_pipeline_from_default_config():
    pipeline_config = get_default_text_pipeline_config()

    import os
    os.environ["ONTOLOGY_PATH"] = "data/input_final/target_kg/ontology.ttl"

    seed_path = Path("data/input_final/target_kg/graph.nt")
    source_path = Path("data/input_final/txt_source/docs")
    result_path = Path("data/tmp/text_pipelines/result.nt")
    tasks_tmp_dir = Path("data/tmp/text_pipelines/tasks_tmp")
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = KgPipe(
        tasks=pipeline_config.tasks, 
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
        name="test_text_pipeline")

    pipeline.build(
        stable_files=True,
        configCatalog=pipeline_config.config_catalog,
        source=Data(path=source_path, format=DataFormat.TEXT), 
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES))

    pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=False)


