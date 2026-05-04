from kgpipe.common import KgPipe, Data, DataFormat
from kgpipe.common.model.configuration import ConfigurationProfile, ParameterBinding, ConfigurationDefinition
from param_opti.tasks.paris import paris_graph_alignment_task
from param_opti.tasks.fusion import fusion_first_value_task
from param_opti.tasks.openie import openie_pipeline_task
from param_opti.tasks.base_linker import relation_linker_label_alias_embedding_transformer_task, entity_linker_label_alias_embedding_transformer_task
from pathlib import Path
from typing import List
import pytest
# Using ground truth

# 1. execute PARIS pipeline, with different thresholds
# 2. evaluate the quality of the pipeline, with different thresholds


# - [ ] impl paris wrapper with exchange and threshold filter

ontology_path = "tmp/ontology.ttl"

tmp_base_dir = Path("data/tmp/rdf_pipelines")
tmp_base_dir.mkdir(parents=True, exist_ok=True)


def _write_to_file(string: str, path: Path):
    with open(path, "w") as f:
        f.write(string)

def _get_param(definition: ConfigurationDefinition, param_name: str):
    params = getattr(definition, "parameters", None)
    if params is None:
        raise KeyError(f"Task config_spec has no parameters field (missing {param_name})")

    if hasattr(params, "get"):
        p = params.get(param_name)
        if p is None:
            raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")
        return p

    for p in params:
        if getattr(p, "name", None) == param_name:
            return p
    raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")


def get_paris_pipeline(entity_matching_threshold: float, relation_matching_threshold: float):
    name = (
        f"paris_graph_alignment(entity={entity_matching_threshold},rel={relation_matching_threshold})"
        "_fusion_first_value"
    )

    seed_path = Path("data/inputs/target_kg/data.nt")
    source_path = Path("data/inputs/rdf_source/data.nt")
    result_path = Path(f"data/tmp/rdf_pipelines/result_{entity_matching_threshold}_{relation_matching_threshold}.nt")
    tasks_tmp_dir = Path(f"data/tmp/rdf_pipelines/tasks_tmp_{entity_matching_threshold}_{relation_matching_threshold}")
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

    config_catalog = {
        "paris_graph_alignment": ConfigurationProfile(
            name=f"paris_graph_alignment_entity={entity_matching_threshold},relation={relation_matching_threshold}",
            definition=paris_graph_alignment_task.config_spec,
            bindings=[
                ParameterBinding(
                    parameter=_get_param(paris_graph_alignment_task.config_spec, "entity_matching_threshold"),
                    value=entity_matching_threshold,
                ),
                ParameterBinding(
                    parameter=_get_param(paris_graph_alignment_task.config_spec, "relation_matching_threshold"),
                    value=relation_matching_threshold,
                ),
            ],
        ),
        "fusion_first_value": ConfigurationProfile(
            name="fusion_first_value",
            definition=fusion_first_value_task.config_spec,
            bindings=[
                ParameterBinding(
                    parameter=_get_param(fusion_first_value_task.config_spec, "ontology_path"),
                    value=ontology_path,
                ),
            ],
        )
    }

    pipeline = KgPipe(
        name=name,
        tasks=[paris_graph_alignment_task, fusion_first_value_task],
        seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
        data_dir=tasks_tmp_dir,
    )

    pipeline.build(
        stable_files=True,
        configCatalog=config_catalog,
        source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )

    return pipeline, config_catalog

def get_openie_pipeline(entity_linking_threshold: float, relation_linking_threshold: float):
    name = (
        f"openie_pipeline(entity={entity_linking_threshold},rel={relation_linking_threshold})"
    )
    
    seed_path = Path("data/inputs/target_kg/data.nt")
    source_path = Path("data/inputs/text_source/docs")
    result_path = Path(f"data/tmp/rdf_pipelines/result_{entity_linking_threshold}_{relation_linking_threshold}.nt")
    tasks_tmp_dir = Path(f"data/tmp/rdf_pipelines/tasks_tmp_{entity_linking_threshold}_{relation_linking_threshold}")
    tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

    config_catalog = {
        "openie_pipeline": ConfigurationProfile(
            name=name,
            definition=openie_pipeline_task.config_spec,
            bindings=[
                ParameterBinding(parameter=_get_param(openie_pipeline_task.config_spec, "entity_linking_threshold"), value=entity_linking_threshold),
                ParameterBinding(parameter=_get_param(openie_pipeline_task.config_spec, "relation_linking_threshold"), value=relation_linking_threshold),
            ],
        ),
    }

    pipeline = KgPipe(
        name=name,
        tasks=[openie_pipeline_task, relation_linker_label_alias_embedding_transformer_task, entity_linker_label_alias_embedding_transformer_task],
        seed=Data(path=seed_path, format=DataFormat.TEXT),
        data_dir=tasks_tmp_dir,
    )

    pipeline.build(
        stable_files=True,
        configCatalog=config_catalog,
        source=Data(path=source_path, format=DataFormat.TEXT),
        result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES),
    )

    return pipeline, config_catalog

# parameterize the test with different thresholds for entity matching and relation matching
@pytest.mark.parametrize("entity_matching_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("relation_matching_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_paris_pipelines(entity_matching_threshold, relation_matching_threshold):
    """
    test a paris pipeline with different thresholds for entity matching and relation matching
    """
    pipeline, config_catalog = get_paris_pipeline(
        entity_matching_threshold, relation_matching_threshold
    )
    pipeline.run(configCatalog=config_catalog, stable_files_override=False)
    print(f"Pipeline run with entity_matching_threshold={entity_matching_threshold} and relation_matching_threshold={relation_matching_threshold}")


@pytest.mark.parametrize("entity_matching_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("relation_matching_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_eval_paris_pipeline(entity_matching_threshold, relation_matching_threshold):
    """
    evaluate a paris pipeline with different thresholds for entity matching and relation matching
    current best "entity_alignment_0.9_0.7" with f1 score 0.971
    """

    print(f"Evaluating triple alignment with entity_matching_threshold={entity_matching_threshold} and relation_matching_threshold={relation_matching_threshold}...")
    from kgpipe_eval.utils.kg_utils import KgManager
    from kgpipe_eval.metrics.triple_alignment import TripleAlignmentMetric, TripleAlignmentConfig
    from kgpipe_eval.metrics.entity_alignment import EntityAlignmentMetric, EntityAlignmentConfig
    from kgpipe_eval.api import MetricResult
    from kgpipe_eval.test.utils import render_metric_result

    ref_kg_path = Path("data/inputs/reference_kg/data_agg.nt")
    gen_kg_path = Path(f"data/tmp/rdf_pipelines/result_{entity_matching_threshold}_{relation_matching_threshold}.nt")

    entity_alignment_config = EntityAlignmentConfig(
            method="label_embedding",
            reference_kg=ref_kg_path,
            verified_entities_path=None,
            verified_entities_delimiter="\t",
            entity_sim_threshold=0.95
    )

    tg = KgManager.load_kg(gen_kg_path)
    metric_result : MetricResult = EntityAlignmentMetric().compute(tg, entity_alignment_config)
    result_string = render_metric_result(metric_result)
    _write_to_file(result_string, Path(f"data/tmp/rdf_pipelines/entity_alignment_{entity_matching_threshold}_{relation_matching_threshold}.txt"))


    triple_alignment_config = TripleAlignmentConfig(
        reference_kg=ref_kg_path,
        entity_alignment_config=entity_alignment_config,
        value_sim_threshold=0.5,
        cache_literal_embeddings=True
    )

    tg = KgManager.load_kg(gen_kg_path)
    metric_result : MetricResult = TripleAlignmentMetric().compute(tg, triple_alignment_config)
    result_string = render_metric_result(metric_result)
    _write_to_file(result_string, Path(f"data/tmp/rdf_pipelines/triple_alignment_{entity_matching_threshold}_{relation_matching_threshold}.txt"))


@pytest.mark.parametrize("entity_linking_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
@pytest.mark.parametrize("relation_linking_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_openie_pipeline(entity_linking_threshold, relation_linking_threshold):
    """
    test the openie pipeline
    """
    pipeline, config_catalog = get_openie_pipeline(entity_linking_threshold, relation_linking_threshold)
    
    print(pipeline.plan())