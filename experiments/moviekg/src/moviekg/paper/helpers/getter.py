
import pandas as pd
from collections import defaultdict
import json
from typing import List, Callable


type pipeline_name = str
type stage_name = str
type metric_name = str
type metric_value = float
type pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]]
type pipeline_stage_metric_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]]

"""
Helper file to map final metrics as kgpipe.evaluation... still in progress

Each getter returns a nested dictionary of pipeline, stage, metric_name
{
    "pipeline": {
        "stage": {
            "metric_name": value
        }
    }
}
"""

# Util

def dict_for_metric_name(df: pd.DataFrame, metric_name: str, row_name: str = "value") -> pipeline_stage_dict:
    df = df[df["metric"] == metric_name]
    metric_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for index, row in df.iterrows():
        metric_dict[row["pipeline"]][row["stage"]] = row[row_name]
    return metric_dict

# Statistical metrics

def sta_entity_count(df: pd.DataFrame):
    # only pipeline, stage, value
    return dict_for_metric_name(df, "entity_count")

def sta_fact_count(df: pd.DataFrame):
    return dict_for_metric_name(df, "triple_count")

def sta_type_count(df: pd.DataFrame):
    return dict_for_metric_name(df, "class_count")

def sta_relation_count(df: pd.DataFrame):
    return dict_for_metric_name(df, "relation_count")

def sta_shallow_entity_count(df: pd.DataFrame):
    return dict_for_metric_name(df, "shallow_entity_count")

def sta_denisity(df: pd.DataFrame):
    fact_count = sta_fact_count(df)
    entity_count = sta_entity_count(df)

    metric_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for pipeline, stage_dict in fact_count.items():
        for stage, value in stage_dict.items():
            metric_dict[pipeline][stage] = value / entity_count[pipeline][stage]

    return metric_dict

def sta_duration(df: pd.DataFrame):
    return dict_for_metric_name(df, "duration")

# def sta_memory_peak(df: pd.DataFrame):
#     return dict_for_metric_name(df, "memory_peak")

# Semantic metrics

def sem_disjoint_domain(df: pd.DataFrame):
    return dict_for_metric_name(df, "disjoint_domain", "normalized")

def sem_incorrect_relation_direction(df: pd.DataFrame):
    return dict_for_metric_name(df, "incorrect_relation_direction", "normalized")

def sem_incorrect_relation_cardinality(df: pd.DataFrame):
    return dict_for_metric_name(df, "incorrect_relation_cardinality", "normalized")

def sem_incorrect_relation_range(df: pd.DataFrame):
    return dict_for_metric_name(df, "incorrect_relation_range", "normalized")

def sem_incorrect_relation_domain(df: pd.DataFrame):
    return dict_for_metric_name(df, "incorrect_relation_domain", "normalized")

def sem_incorrect_datatype(df: pd.DataFrame):
    return dict_for_metric_name(df, "incorrect_datatype", "normalized")

def sem_incorrect_datatype_format(df: pd.DataFrame):
    return dict_for_metric_name(df, "incorrect_datatype_format", "normalized")

# Reference metrics
def ref_kg_f1(df: pd.DataFrame):
    df = df[df["metric"] == "ReferenceTripleAlignmentMetricSoftEV"]

    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        # print(details)
        f1 = details.get("f1_score", -1)
        res[row.pipeline][row.stage] = f1
    return res

def ref_kg_p(df: pd.DataFrame):
    df = df[df["metric"] == "ReferenceTripleAlignmentMetricSoftEV"]

    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        # print(details)
        p = details["precision"]
        res[row.pipeline][row.stage] = p
    return res

def ref_kg_r(df: pd.DataFrame):
    df = df[df["metric"] == "ReferenceTripleAlignmentMetricSoftE"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        # print(details)
        r = details["recall"]
        res[row.pipeline][row.stage] = r
    return res

def ref_source_entity_f1(df: pd.DataFrame) -> pipeline_stage_dict:
    df = df[df["metric"] == "SourceEntityPrecisionMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        expected_entities_count = details["expected_entities_count"]
        found_entities_count = details["found_entities_count"]
        overlapping_entities_count = details["overlapping_entities_count"]
        possible_duplicates_count = details["possible_duplicates_count"]
        overlapping_entities_strict_count = details["overlapping_entities_strict_count"]
        
        precision = overlapping_entities_strict_count / overlapping_entities_count
        precision = precision if precision <= 1.0 else 1.0
        recall = overlapping_entities_count / expected_entities_count
        recall = recall if recall <= 1.0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall)
        df.loc[row.Index, "normalized"] = f1
        res[row.pipeline][row.stage] = f1
    return res

def ref_source_entity_p(df: pd.DataFrame):
    df = df[df["metric"] == "SourceEntityPrecisionMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        precision = details["overlapping_entities_strict_count"] / details["overlapping_entities_count"]
        res[row.pipeline][row.stage] = precision
    return res

def ref_source_entity_r(df: pd.DataFrame):
    df = df[df["metric"] == "SourceEntityPrecisionMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        recall = details["overlapping_entities_count"] / details["expected_entities_count"]
        res[row.pipeline][row.stage] = recall
    return res

def ref_entity_matching_f1(df: pd.DataFrame):
    df = df[df["metric"] == "ER_EntityMatchMetric"]

    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_seed_match_cnt"]
        fp = details["false_seed_match_cnt"]
        fn = details["false_missing_seed_match_cnt"]
        f1 = 2 * tp / (2 * tp + fp + fn)
        res[row.pipeline][row.stage] = f1
    return res

def ref_entity_matching_p(df: pd.DataFrame):
    df = df[df["metric"] == "ER_EntityMatchMetric"]

    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_seed_match_cnt"]
        fp = details["false_seed_match_cnt"]
        fn = details["false_missing_seed_match_cnt"]
        precision = tp / (tp + fp)
        res[row.pipeline][row.stage] = precision
    return res

def ref_entity_matching_r(df: pd.DataFrame):
    df = df[df["metric"] == "ER_EntityMatchMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_seed_match_cnt"]
        fp = details["false_seed_match_cnt"]
        fn = details["false_missing_seed_match_cnt"]
        recall = tp / (tp + fn)
        res[row.pipeline][row.stage] = recall
    return res

RM_DEFAULT_FN=24 # 23 + label

def ref_relation_matching_f1(df: pd.DataFrame):
    df = df[df["metric"] == "ER_RelationMatchMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_relation_match_cnt"]
        fp = details["false_relation_match_cnt"]
        fn = RM_DEFAULT_FN - (tp+fp) # details.get("false_missing_relation_match_cnt", 0)
        f1 = 2 * tp / (2 * tp + fp + fn)
        res[row.pipeline][row.stage] = f1
    return res


def ref_relation_matching_p(df: pd.DataFrame):
    df = df[df["metric"] == "ER_RelationMatchMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_relation_match_cnt"]
        fp = details["false_relation_match_cnt"]
        fn = RM_DEFAULT_FN - (tp+fp) # details.get("false_missing_relation_match_cnt", 0)
        print(f"tp, fp, fn for {row.pipeline} {row.stage}: {tp}, {fp}, {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        res[row.pipeline][row.stage] = precision
    return res


def ref_relation_matching_r(df: pd.DataFrame):
    df = df[df["metric"] == "ER_RelationMatchMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_relation_match_cnt"]
        fp = details["false_relation_match_cnt"]
        fn = RM_DEFAULT_FN - (tp+fp) # details.get("false_missing_relation_match_cnt", 0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        res[row.pipeline][row.stage] = recall
    return res

def ref_entity_linking_r(df: pd.DataFrame):
    df = df[df["metric"] == "TE_ExpectedEntityLinkMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        # print(details)
        tp = details["true_link_cnt"]
        fp = details["false_link_cnt"]
        fn = details["false_missing_link_cnt"]
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        res[row.pipeline][row.stage] = r
    return res

def ref_json_entity_matching_f1(df: pd.DataFrame):
    df = df[df["metric"] == "JsonEntityMatchingMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        res[row.pipeline][row.stage] = details["f1_score"]
    return res

def ref_json_entity_matching_p(df: pd.DataFrame):
    df = df[df["metric"] == "JsonEntityMatchingMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        res[row.pipeline][row.stage] = details["precision"]
    return res

def ref_json_entity_matching_r(df: pd.DataFrame):
    df = df[df["metric"] == "JsonEntityMatchingMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        res[row.pipeline][row.stage] = details["recall"]
    return res

def ref_json_entity_linking_r(df: pd.DataFrame):
    df = df[df["metric"] == "JsonEntityLinkingMetric"]
    res: pipeline_stage_dict = defaultdict[pipeline_name, defaultdict[stage_name, metric_value]](lambda: defaultdict[stage_name, metric_value](lambda: None))
    for row in df.itertuples():
        details = json.loads(row.details)
        if "error" in details:
            res[row.pipeline][row.stage] = -1
            continue
        res[row.pipeline][row.stage] = details["recall"]
    return res

TABLE_DISPLAY_NAMES = {
    # Statistical metrics
    sta_entity_count.__name__   : "EC",
    sta_fact_count.__name__: "FC",
    sta_type_count.__name__: "TC",
    sta_relation_count.__name__: "RC",
    sta_shallow_entity_count.__name__: "SEC",
    sta_denisity.__name__: "D",
    sta_duration.__name__: "T",
    # sta_memory_peak.__name__: "M",
    # Semantic metrics
    sem_disjoint_domain.__name__: "ODT",
    sem_incorrect_relation_direction.__name__: "ORD",
    sem_incorrect_relation_cardinality.__name__: "OCA",
    sem_incorrect_relation_range.__name__: "OR",
    sem_incorrect_relation_domain.__name__: "OD",
    sem_incorrect_datatype.__name__: "OLT",
    sem_incorrect_datatype_format.__name__: "OLF",
    # Reference metrics
    ref_kg_f1.__name__: "RTC",
    ref_kg_p.__name__: "RTC-SoftE",
    ref_kg_r.__name__: "RTC-SoftE-R",
    ref_source_entity_f1.__name__: "VSEC",
    ref_source_entity_p.__name__: "VSEC-P",
    ref_source_entity_r.__name__: "VSEC-R",
    ref_entity_matching_f1.__name__: "ER-EM",
    ref_entity_matching_p.__name__: "ER-EM-P",
    ref_entity_matching_r.__name__: "ER-EM-R",
    ref_relation_matching_f1.__name__: "ER-RM",
    ref_relation_matching_p.__name__: "ER-RM-P",
    ref_relation_matching_r.__name__: "ER-RM-R",
    ref_entity_linking_r.__name__: "TE-EEL",
    ref_json_entity_matching_f1.__name__: "JSON-EM",
    ref_json_entity_matching_p.__name__: "JSON-EM-P",
    ref_json_entity_matching_r.__name__: "JSON-EM-R",
    ref_json_entity_linking_r.__name__: "JSON-EL",
}

def dict_of_metrics(df: pd.DataFrame, metric_getters: List[Callable[[pd.DataFrame], dict]]) -> pipeline_stage_metric_dict:
    """
    # call the getter functions for each metric name not the dict_for_metric_name
    """

    # Create a 3-level nested defaultdict: pipeline -> stage -> metric_name -> value
    metric_dict = defaultdict(lambda: defaultdict(dict))

    for metric_getter in metric_getters:
        metric_name = metric_getter.__name__          # the metric name (e.g., "sta_entity_count")
        metric_data = metric_getter(df)               # returns pipeline->stage->value

        if metric_data is None:
            continue

        for pipeline, stage_dict in metric_data.items():
            for stage, value in stage_dict.items():
                metric_dict[pipeline][stage][metric_name] = value

    return metric_dict

def get_pipeline_stage_metric_dict(df: pd.DataFrame, metric_names: List[str]) -> pipeline_stage_metric_dict:
    """
    # call the getter functions for each metric name not the dict_for_metric_name
    """    
    return dict_of_metrics(df, [globals()[f"{metric_name.lower()}"] for metric_name in TABLE_DISPLAY_NAMES.keys()])


def normalize_min_best(values: List[float], value: float) -> float:
    def norm_min(min, value):
        return (min/value)
    return norm_min(min(values), value)


def normalize_max_best(values: List[float], value: float) -> float:
    # print(f"values: {values}, value: {value}")
    def norm_max(max, value):
        return 1 / (max/value)
    return norm_max(max(values), value)


def normalize_metric(psmd: pipeline_stage_metric_dict, metric_name: str, stages: List[str], func: Callable[[list[float], float], float]) -> pipeline_stage_metric_dict:
    values_for_metric = []

    pipelines_to_normalize = []
    stages_to_normalize = []


    for pipeline, stage_dict in psmd.items():
        pipelines_to_normalize.append(pipeline)
        for stage, metric_dict in stage_dict.items():
            if stage not in stages or metric_name not in metric_dict:
                continue
            stages_to_normalize.append(stage)
            values_for_metric.append(metric_dict[metric_name])
                
    for pipeline in pipelines_to_normalize:
        for stage in stages_to_normalize:
            if metric_name not in psmd[pipeline][stage]:
                continue
            value = psmd[pipeline][stage][metric_name]
            if metric_name == sta_fact_count.__name__:
                if pipeline in ["json_llm_mapping_v1", "text_llm_triple_extract_v1"]:
                    values_for_metric=[65000]
                else:
                    values_for_metric=[340000]
                print(f"setting max ec for norm {pipeline} {values_for_metric}")
            psmd[pipeline][stage][metric_name+"_norm"] = func(values_for_metric, value)

    return psmd

def update_task_selected_task_metric(psmd: pipeline_stage_metric_dict, metric_name: str) -> pipeline_stage_metric_dict:
    
    for pipleine, stage_dict in psmd.items():
        if pipleine in ["reference", "seed"]:
            continue
        for stage, metric_dict in stage_dict.items():
            entity_matching_f1 = metric_dict.get(ref_entity_matching_f1.__name__, -1)
            relation_matching_f1 = metric_dict.get(ref_relation_matching_f1.__name__, -1)
            entity_linking_r = metric_dict.get(ref_entity_linking_r.__name__, -1)
            json_entity_matching_f1 = metric_dict.get(ref_json_entity_matching_f1.__name__, -1)
            json_entity_linking_r = metric_dict.get(ref_json_entity_linking_r.__name__, -1)

            if json_entity_matching_f1 != -1:
                metric_dict[metric_name] = json_entity_matching_f1
                metric_dict[metric_name+"_spec"] = "JSON ER"
            elif entity_matching_f1 != -1:
                metric_dict[metric_name] = (entity_matching_f1 + relation_matching_f1) / 2
                metric_dict[metric_name+"_spec"] = "RDF ER"
            elif json_entity_linking_r != -1:
                metric_dict[metric_name] = json_entity_linking_r
                metric_dict[metric_name+"_spec"] = "JSON EL"
            else:
                metric_dict[metric_name] = entity_linking_r
                metric_dict[metric_name+"_spec"] = "TE"
    
    return psmd

def agg_avg(values: list[float]) -> float:
    return sum(values) / len(values)

def agg_sum(values: list[float]) -> float:
    return sum(values)

def agg_metric_over_stages(psmd: pipeline_stage_metric_dict, metric_name: str, suffix: str, agg_func: Callable[[list[float]], float]) -> pipeline_stage_metric_dict:

    values_for_metric_by_pipeline = defaultdict[pipeline_name, list[float]](lambda: [])
    pipelines_to_agg = []
    stages_to_agg = []

    for pipeline, stage_dict in psmd.items():
        if pipeline in ["reference", "seed"]:
            continue
        pipelines_to_agg.append(pipeline)
        for stage, metric_dict in stage_dict.items():
            if metric_name not in metric_dict:
                continue
            stages_to_agg.append(stage)
            values_for_metric_by_pipeline[pipeline].append(metric_dict[metric_name])

    for pipeline in pipelines_to_agg:
        try:
            psmd[pipeline]["stage_3"][metric_name+suffix] = agg_func(values_for_metric_by_pipeline[pipeline])
        except Exception as e:
            print(f"Error aggregating metric {metric_name} for pipeline {pipeline}: {e}")
            print(values_for_metric_by_pipeline[pipeline])
            psmd[pipeline]["stage_3"][metric_name+suffix] = 0
    return psmd

def apply_selected_updates(psmd: pipeline_stage_metric_dict) -> pipeline_stage_metric_dict:
    normalize_metric(psmd, "sta_entity_count", ["stage_3"], normalize_max_best)
    update_task_selected_task_metric(psmd, "ref_selected_task_metric")
    agg_metric_over_stages(psmd, "ref_selected_task_metric", "_avg", agg_avg)
    agg_metric_over_stages(psmd, "sta_duration", "_sum", agg_sum)
    agg_metric_over_stages(psmd, "ref_source_entity_f1", "_avg", agg_avg)
    agg_metric_over_stages(psmd, "ref_kg_f1", "_avg", agg_avg)
    return psmd

def test_getter():
    from pathlib import Path
    from moviekg.paper.helpers.helpers import load_metrics_from_file
    print(TABLE_DISPLAY_NAMES.keys())
    df = load_metrics_from_file(Path("/home/marvin/project/data/out/large") / "all_metrics.csv")
    psmd = dict_of_metrics(df, [globals()[f"{metric_name.lower()}"] for metric_name in TABLE_DISPLAY_NAMES.keys()])

    
    apply_selected_updates(psmd)

    for pipeline, stage_dict in psmd.items():
        for stage, metric_dict in stage_dict.items():
            if pipeline in ["reference", "seed"]:
                continue
            # print(f"{pipeline} {stage} {metric_dict['ref_selected_task_metric']} {metric_dict['ref_selected_task_metric_spec']}")
            if "stage_3" == stage:
                # print(f"{pipeline} {stage} {metric_dict['ref_selected_task_metric_agg']}")
                print(pipeline)
                print(stage)
                print(json.dumps(metric_dict, indent=4))
                print("--------------------------------")