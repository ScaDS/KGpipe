from typing import List, Dict, Any, Optional
import json
import random
from kgpipe.common import KgPipe, Data, DataFormat, Registry
from kgpipe.common.model.configuration import ConfigurationProfile, ParameterBinding
from kgpipe.common.model.task import KgTask
from pydantic import BaseModel

from param_opti.tasks.paris import paris_graph_alignment_task, paris_entity_alignment_task, paris_ontology_matching_task
from param_opti.tasks.fusion import fusion_first_value_task
from param_opti.tasks.base_linker import relation_linker_label_alias_embedding_transformer_task, entity_linker_label_alias_embedding_transformer_task
from param_opti.tasks.base_matcher import (
    graph_alignment_label_alias_embedding_transformer_task,
    relation_matcher_label_alias_embedding_transformer_task,
    entity_matcher_label_alias_embedding_transformer_task,
)
from param_opti.tasks.corenlp import corenlp_text_extraction_task
from param_opti.tasks.genie import genie_text_extraction_task
from param_opti.tasks.spotlight import spotlight_entity_linking_task
from param_opti.tasks.matching_helpers import aggregate_matching_results_task

from kgpipe.generation.loaders import build_from_conf
from pathlib import Path
# for given tasks and config parameters, generate a pipeline (KGpipe)

tmp_base_dir = Path("tmp")
if not tmp_base_dir.exists():
    tmp_base_dir.mkdir(parents=True, exist_ok=True)

RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "rdf_sampled_pipeline_configs.json"
_RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION = 1


class PipelineConfig(BaseModel):
    tasks: List[KgTask]
    config_catalog: Dict[str, ConfigurationProfile]

RDF_SEARCH_SPACE = {
    "graph_alignment_label_alias_embedding_transformer_task": {
        "category": ["ontology_matching", "entity_matching", "aggregate_matching_results"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
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
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_linker_label_alias_embedding_transformer_task": {
        "category": "entity_linking",
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
}

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
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "entity_linker_label_alias_embedding_transformer_task": {
        "category": ["entity_linking"],
        "model_name": ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "fusion_first_value_task": {
        "category": ["fusion"],
    },
}

TEXT_TASK_DICT = {
    "corenlp_text_extraction_task": corenlp_text_extraction_task,
    "genie_text_extraction_task": genie_text_extraction_task,
    "spotlight_entity_linking_task": spotlight_entity_linking_task,
    "relation_linker_label_alias_embedding_transformer_task": relation_linker_label_alias_embedding_transformer_task,
    "entity_linker_label_alias_embedding_transformer_task": entity_linker_label_alias_embedding_transformer_task,
    "fusion_first_value_task": fusion_first_value_task,
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

for task_name, task in RDF_TASK_DICT.items():
    Registry.add_task(task_name, task)

class PipelineLayout(BaseModel):
    """
    allowed task categories in the pipeline
    """
    allowed_task_categories: List[str]


def _task_categories_list(search_space: Dict[str, Dict[str, Any]], task_name: str) -> List[str]:
    raw = search_space.get(task_name, {}).get("category")
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, str)]
    if isinstance(raw, str):
        return [raw]
    return []


def enumerate_valid_task_combinations(
    search_space: Dict[str, Dict[str, Any]],
    pipeline_layout: PipelineLayout,
) -> List[List[str]]:
    """
    Enumerate all possible task-name combinations for the given pipeline layout,
    respecting category order and multi-category coverage, without sampling config options.

    A task is only eligible for the current category if its declared categories are
    disjoint from categories already covered by earlier tasks. That avoids pairing e.g.
    Paris ontology matching with a dual-category embedding matcher that would repeat
    ontology coverage when only entity matching is still needed.
    """
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

            eligible: List[str] = []
            for tn in all_task_names:
                cats = _task_categories_list(search_space, tn)
                if category not in cats:
                    continue
                if set(cats) & covered:
                    continue
                eligible.append(tn)
            for tn in eligible:
                new_combo = combo + [tn]
                new_covered = set(covered)
                new_covered.update(_task_categories_list(search_space, tn))
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


def pipeline_config_to_snapshot(task_keys: List[str], pipeline_config: PipelineConfig) -> Dict[str, Any]:
    profiles: Dict[str, Any] = {}
    for task in pipeline_config.tasks:
        prof = pipeline_config.config_catalog.get(task.name)
        if prof is None:
            continue
        profiles[task.name] = {
            "profile_name": prof.name,
            "bindings": [
                {"parameter": binding.parameter.name, "value": binding.value}
                for binding in prof.bindings
            ],
        }
    return {"task_keys": task_keys, "profiles": profiles}


def pipeline_config_from_snapshot(snapshot: Dict[str, Any]) -> PipelineConfig:
    task_keys: List[str] = snapshot["task_keys"]
    profiles: Dict[str, Any] = snapshot.get("profiles") or {}
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}

    for task_key in task_keys:
        task = task_dict[task_key]
        tasks.append(task)
        prof_data = profiles.get(task.name)
        if prof_data is None:
            continue
        if getattr(task, "config_spec", None) is None:
            continue
        bindings = [
            ParameterBinding(
                parameter=_get_param(task.config_spec, b["parameter"]),
                value=b["value"],
            )
            for b in prof_data["bindings"]
        ]
        config_catalog[task.name] = ConfigurationProfile(
            name=prof_data["profile_name"],
            definition=task.config_spec,
            bindings=bindings,
        )

    return PipelineConfig(tasks=tasks, config_catalog=config_catalog)


def load_rdf_sampled_pipeline_configs(path: Optional[Path] = None) -> List[PipelineConfig]:
    fixture_path = path or RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    if raw.get("version") != _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION:
        raise ValueError(
            f"Unsupported rdf sampled configs snapshot version {raw.get('version')!r}; "
            f"expected {_RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION}"
        )
    return [pipeline_config_from_snapshot(item) for item in raw["samples"]]


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

        eligible_task_names = [
            tn
            for tn in eligible_task_names
            if not (set(_task_categories_list(search_space, tn)) & covered_categories)
        ]
        if not eligible_task_names:
            raise ValueError(
                f"No task can cover category {category!r} without overlapping already covered "
                f"categories {sorted(covered_categories)}. Adjust search_space or pipeline_layout."
            )

        task_key = random.choice(eligible_task_names)
        task = task_dict[task_key]
        covered_categories.update(_task_categories_list(search_space, task_key))
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

def sample_config_catalog_for_task_combo(
    search_space: Dict[str, Dict[str, Any]],
    task_name_combo: List[str],
    *,
    rng: random.Random,
) -> PipelineConfig:
    tasks: List[KgTask] = []
    config_catalog: Dict[str, ConfigurationProfile] = {}

    for task_key in task_name_combo:
        task = task_dict[task_key]
        tasks.append(task)

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

            config_value = rng.choice(config_values)
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



def test_sample_valid_rdf_pipeline_config():
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["ontology_matching", "entity_matching", "aggregate_matching_results", "fusion"]
    )
    pipeline_config = sample_valid_pipeline_config(RDF_SEARCH_SPACE, pipeline_layout)
    print_pipeline_config_short(pipeline_config)

def test_enumerate_all_valid_rdf_task_combinations_no_config_sampling():
    print("enumerate_all_valid_rdf_task_combinations_no_config_sampling")
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["ontology_matching", "entity_matching", "aggregate_matching_results", "fusion"]
    )
    combos = enumerate_valid_task_combinations(RDF_SEARCH_SPACE, pipeline_layout)

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

def test_enumerate_all_valid_rdf_task_combinations_with_config_sampling():
    print("enumerate_all_valid_rdf_task_combinations_with_config_sampling")
    n = 1
    rng = random.Random(0)

    pipeline_layout = PipelineLayout(
        allowed_task_categories=["ontology_matching", "entity_matching", "aggregate_matching_results", "fusion"]
    )
    combos = enumerate_valid_task_combinations(RDF_SEARCH_SPACE, pipeline_layout)

    total_config_count = 0
    snapshots: List[Dict[str, Any]] = []

    for combo in combos:
        print()
        print("combo:", combo)
        for i in range(n):
            total_config_count += 1
            print(f"sample {total_config_count}/{len(combos) * n}")
            pipeline_config = sample_config_catalog_for_task_combo(
                RDF_SEARCH_SPACE, combo, rng=rng
            )

            print_pipeline_config_short(pipeline_config)
            snapshots.append(pipeline_config_to_snapshot(combo, pipeline_config))

    RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    RDF_SAMPLED_PIPELINE_CONFIGS_FIXTURE.write_text(
        json.dumps(
            {"version": _RDF_PIPELINE_CONFIG_SNAPSHOT_VERSION, "samples": snapshots},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_sample_valid_text_pipeline_config():
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["information_extraction", "entity_linking", "relation_linking", "fusion"]
    )
    pipeline_config = sample_valid_pipeline_config(TEXT_SEARCH_SPACE, pipeline_layout)
    print_pipeline_config_short(pipeline_config)


def test_enumerate_all_valid_text_task_combinations_no_config_sampling():
    print("enumerate_all_valid_text_task_combinations_no_config_sampling")
    pipeline_layout = PipelineLayout(
        allowed_task_categories=["information_extraction", "entity_linking", "relation_linking", "fusion"]
    )
    combos = enumerate_valid_task_combinations(TEXT_SEARCH_SPACE, pipeline_layout)
    for combo in combos:
        print(combo)

def test_enumerate_all_valid_text_task_combinations_with_config_sampling():
    print("enumerate_all_valid_text_task_combinations_with_config_sampling")
    n = 5
    rng = random.Random(0)

    pipeline_layout = PipelineLayout(
        allowed_task_categories=["information_extraction", "entity_linking", "relation_linking", "fusion"]
    )
    combos = enumerate_valid_task_combinations(TEXT_SEARCH_SPACE, pipeline_layout)

    total_config_count = 0

    for combo in combos:
        print()
        print("combo:", combo)
        for i in range(n):
            total_config_count += 1
            print(f"sample {total_config_count}/{len(combos) * n}")
            pipeline_config = sample_config_catalog_for_task_combo(
                TEXT_SEARCH_SPACE, combo, rng=rng
            )
            print_pipeline_config_short(pipeline_config)



# def test_rdf_pipeline_from_config():
#     pipeline_config = sample_valid_pipeline_config(RDF_SEARCH_SPACE, PipelineLayout(allowed_task_categories=["entity_matching", "fusion"]))

#     seed_path = tmp_base_dir / "seed.nt"
#     source_path = tmp_base_dir / "source.nt"
#     result_path = tmp_base_dir / "result.nt"
#     tasks_tmp_dir = tmp_base_dir / "tasks_tmp"
#     tasks_tmp_dir.mkdir(parents=True, exist_ok=True)

#     # Ensure inputs exist for pipeline execution.
#     seed_path.write_text("<http://example.org/s> <http://example.org/p> <http://example.org/o> .\n")
#     source_path.write_text("<http://example.org/s2> <http://example.org/p> <http://example.org/o> .\n")

#     pipeline = KgPipe(
#         tasks=pipeline_config.tasks, 
#         seed=Data(path=seed_path, format=DataFormat.RDF_NTRIPLES),
#         data_dir=tasks_tmp_dir,
#         name="test_pipeline")

#     pipeline.build(
#         stable_files=True,
#         configCatalog=pipeline_config.config_catalog,
#         source=Data(path=source_path, format=DataFormat.RDF_NTRIPLES), 
#         result=Data(path=result_path, format=DataFormat.RDF_NTRIPLES))

#     pipeline.run(configCatalog=pipeline_config.config_catalog, stable_files_override=True)