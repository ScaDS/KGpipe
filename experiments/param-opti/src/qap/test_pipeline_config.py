from typing import List, Dict, Any, Optional
import random
from kgpipe.common import KgPipe, Data, DataFormat, Registry
from kgpipe.common.model.configuration import ConfigurationProfile, ParameterBinding
from kgpipe.common.model.task import KgTask
from pydantic import BaseModel
from param_opti.tasks.paris import paris_graph_alignment_task, paris_entity_alignment_task, paris_ontology_matching_task
from param_opti.tasks.fusion import fusion_first_value_task
from param_opti.tasks.base_linker import relation_linker_label_alias_embedding_transformer_task, entity_linker_label_alias_embedding_transformer_task
from param_opti.tasks.base_matcher import relation_matcher_label_alias_embedding_transformer_task, entity_matcher_label_alias_embedding_transformer_task
from kgpipe.generation.loaders import build_from_conf
from pathlib import Path
# for given tasks and config parameters, generate a pipeline (KGpipe)

tmp_base_dir = Path("tmp")
if not tmp_base_dir.exists():
    tmp_base_dir.mkdir(parents=True, exist_ok=True)


class PipelineConfig(BaseModel):
    tasks: List[KgTask]
    config_catalog: Dict[str, ConfigurationProfile]

SEARCH_SPACE = {
    "relation_matcher_label_alias_embedding_transformer_task": {
        "category": ["ontology_matching"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_matcher_label_alias_embedding_transformer_task": {
        "category": ["entity_matching"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
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
        "category": ["ontology_matching", "entity_matching"],
        "entity_matching_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
        "relation_matching_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "fusion_first_value_task": {
        "category": ["fusion"],
        # "fusion_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "relation_linker_label_alias_embedding_transformer_task": {
        "category": ["entity_linking"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_linker_label_alias_embedding_transformer_task": {
        "category": "entity_linking",
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
}

task_dict = {
    "relation_matcher_label_alias_embedding_transformer_task": relation_matcher_label_alias_embedding_transformer_task,
    "entity_matcher_label_alias_embedding_transformer_task": entity_matcher_label_alias_embedding_transformer_task,
    "paris_ontology_matching_task": paris_ontology_matching_task,
    "paris_entity_alignment_task": paris_entity_alignment_task,
    "paris_graph_alignment_task": paris_graph_alignment_task,
    "fusion_first_value_task": fusion_first_value_task,
    "relation_linker_label_alias_embedding_transformer_task": relation_linker_label_alias_embedding_transformer_task,
    "entity_linker_label_alias_embedding_transformer_task": entity_linker_label_alias_embedding_transformer_task,
}

for task_name, task in task_dict.items():
    Registry.add_task(task_name, task)

class PipelineLayout(BaseModel):
    """
    allowed task categories in the pipeline
    """
    allowed_task_categories: List[str]

def enumerate_valid_task_combinations(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> List[List[str]]:
    """
    Enumerate all possible task-name combinations for the given pipeline layout,
    respecting category order and multi-category coverage, without sampling config options.
    """
    def _task_categories(task_name: str) -> List[str]:
        raw = search_space.get(task_name, {}).get("category")
        if isinstance(raw, list):
            return [c for c in raw if isinstance(c, str)]
        if isinstance(raw, str):
            return [raw]
        return []

    all_task_names = list(search_space.keys())

    combos: List[List[str]] = [[]]
    covered_sets: List[set[str]] = [set()]

    for category in pipeline_layout.allowed_task_categories:
        next_combos: List[List[str]] = []
        next_covered_sets: List[set[str]] = []

        for combo, covered in zip(combos, covered_sets):
            if category in covered:
                next_combos.append(combo)
                next_covered_sets.append(covered)
                continue

            eligible = [
                tn for tn in all_task_names if category in set(_task_categories(tn))
            ]
            for tn in eligible:
                new_combo = combo + [tn]
                new_covered = set(covered)
                new_covered.update(_task_categories(tn))
                next_combos.append(new_combo)
                next_covered_sets.append(new_covered)

        combos, covered_sets = next_combos, next_covered_sets

    # De-duplicate while keeping stable order.
    seen: set[tuple[str, ...]] = set()
    unique: List[List[str]] = []
    for c in combos:
        t = tuple(c)
        if t in seen:
            continue
        seen.add(t)
        unique.append(c)
    return unique

def _get_param(definition: Any, param_name: str):
    params = getattr(definition, "parameters", None)
    if params is None:
        raise KeyError(f"Task config_spec has no parameters field (missing {param_name})")

    # common shapes: dict-like or list of Parameter
    if hasattr(params, "get"):
        p = params.get(param_name)
        if p is None:
            raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")
        return p

    for p in params:
        if getattr(p, "name", None) == param_name:
            return p
    raise KeyError(f"Parameter {param_name} not found in config_spec.parameters")

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

# TODO rules for valid pipeline config:
def sample_valid_pipeline_config(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> PipelineConfig:
    """
    Randomly sample a valid pipeline config from the search space,
    respecting the order of categories in the pipeline layout.
    """
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}
    covered_categories: set[str] = set()

    for category in pipeline_layout.allowed_task_categories:
        if category in covered_categories:
            continue

        eligible_task_names = [
            tn
            for tn, space in search_space.items()
            if (
                space.get("category") == category
                or (
                    isinstance(space.get("category"), list)
                    and category in (space.get("category") or [])
                )
            )
        ]
        if not eligible_task_names:
            continue

        task_key = random.choice(eligible_task_names)
        task = task_dict[task_key]
        task_categories = search_space.get(task_key, {}).get("category")
        if isinstance(task_categories, list):
            covered_categories.update([c for c in task_categories if isinstance(c, str)])
        elif isinstance(task_categories, str):
            covered_categories.add(task_categories)
        tasks.append(task)

        # metadata only or task has no config spec
        if getattr(task, "config_spec", None) is None:
            continue

        bindings: List[ParameterBinding] = []
        name_parts: List[str] = []
        for config_name, config_values in search_space[task_key].items():
            if config_name == "category":
                continue
            if not isinstance(config_values, list):
                raise TypeError(
                    f"Search space values must be lists; got {task_key}.{config_name}={type(config_values)}"
                )
            if not config_values:
                raise ValueError(f"Empty search space for {task_key}.{config_name}")

            config_value = random.choice(config_values)
            name_parts.append(f"{config_name}={config_value}")
            bindings.append(
                ParameterBinding(
                    parameter=_get_param(task.config_spec, config_name),
                    value=config_value,
                )
            )

        if bindings:
            config_catalog[task.name] = ConfigurationProfile(
                name=f"{task.name}_" + ",".join(name_parts),
                definition=task.config_spec,
                bindings=bindings,
            )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)

def print_pipeline_config_short(pipeline_config: PipelineConfig):
    """
    print the pipeline config in a short format
    """
    print()
    print("================")
    for task in pipeline_config.tasks:
        task_name = task.name
        profile: Optional[ConfigurationProfile] = pipeline_config.config_catalog.get(task_name)
        if profile is None:
            print(f"- {task_name}")
            continue

        parts: List[str] = []
        for binding in profile.bindings:
            parts.append(f"{binding.parameter.name}={binding.value}")
        params = ", ".join(parts)
        print(f"- {task_name}({params})")



def test_sample_valid_rdf_pipeline_config():
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["ontology_matching", "entity_matching", "fusion"]
    )
    pipeline_config = sample_valid_pipeline_config(SEARCH_SPACE, pipeline_layout)
    print_pipeline_config_short(pipeline_config)

def test_enumerate_all_valid_rdf_task_combinations_no_config_sampling():
    print("enumerate_all_valid_rdf_task_combinations_no_config_sampling")
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["ontology_matching", "entity_matching", "fusion"]
    )
    combos = enumerate_valid_task_combinations(SEARCH_SPACE, pipeline_layout)

    for combo in combos:
        print(combo)

    # With current SEARCH_SPACE:
    # - ontology_matching can be satisfied by paris_ontology_matching_task, paris_entity_alignment_task, paris_graph_alignment_task
    # - entity_matching can be satisfied by paris_entity_alignment_task, paris_graph_alignment_task (and may be skipped if already covered)
    # - fusion must be satisfied by fusion_first_value_task
    # expected = {
    #     ("paris_ontology_matching_task", "paris_entity_alignment_task", "fusion_first_value_task"),
    #     ("paris_ontology_matching_task", "paris_graph_alignment_task", "fusion_first_value_task"),
    #     ("paris_graph_alignment_task", "fusion_first_value_task"),
    # }

    # assert set(tuple(c) for c in combos) == expected

def test_sample_valid_text_pipeline_config():
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["information_extraction", "entity_linking", "fusion"]
    )
    pipeline_config = sample_valid_pipeline_config(SEARCH_SPACE, pipeline_layout)
    print_pipeline_config_short(pipeline_config)


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

def test_rdf_pipeline_from_config():
    pipeline_config = sample_valid_pipeline_config(SEARCH_SPACE, PipelineLayout(allowed_task_categories=["entity_matching", "fusion"]))

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