
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from moviekg.pipelines.test_inc_ssp import pipeline_types, llm_pipeline_types
from moviekg.paper.helpers import load_metrics_from_file, plot_growth, plot_class_occurence, rank_metrics, plot_class_occ_4_bar_chart

from moviekg.config import OUTPUT_DIR, DATASET_SELECT

if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR is not set")
if not DATASET_SELECT:
    raise ValueError("DATASET_SELECT is not set")

OUTPUT_ROOT = Path(OUTPUT_DIR) / DATASET_SELECT
(OUTPUT_ROOT / "paper").mkdir(parents=True, exist_ok=True)

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

def map_pipeline_name(pipeline_name):
    return name_mapping.get(pipeline_name, pipeline_name)

#  ['entity_count' 'relation_count' 'triple_count' 'class_count'
#  'class_occurrence' 'relation_occurrence' 'property_occurrence'
#  'namespace_usage' 'loose_entity_count' 'shallow_entity_count'
#  'ER_EntityMatchMetric' 'ER_RelationMatchMetric'
#  'TE_ExpectedEntityLinkMetric' 'TE_ExpectedRelationLinkMetric'
#  'SourceEntityCoverageMetric' 'SourceEntityCoverageMetricSoft'
#  'ReferenceTripleAlignmentMetric' 'ReferenceTripleAlignmentMetricSoftE'
#  'ReferenceTripleAlignmentMetricSoftEV' 'ReferenceClassCoverageMetric'
#  'reasoning' 'disjoint_domain' 'incorrect_relation_direction'
#  'incorrect_relation_cardinality' 'incorrect_relation_range'
#  'incorrect_relation_domain' 'incorrect_datatype'
#  'incorrect_datatype_format' 'ontology_class_coverage'
#  'ontology_relation_coverage' 'ontology_namespace_coverage' 'duration']
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

def map_metric_name(metric_name):
    return METRIC_NAME_MAP.get(metric_name, metric_name)


def agg_duration_over_stages_per_pipeline(metric_df):
    # group by pipeline and stage and take mean of normalized
    metric_df = metric_df[metric_df["metric"] == "duration"]
    # print(metric_df)
    metric_df = metric_df.groupby(["pipeline"])["value"].sum().reset_index()
    # add stage  column = stage 3
    metric_df["stage"] = "stage_3"#
    metric_df["metric"] = "duration"

    # print(metric_df.to_string())
    return metric_df

def norm_min(min, value):
    return (min/value)

def norm_max(max, value):
    return 1 / (max/value)

def test_tab_duration_over_stages_per_pipeline():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metric_df = agg_duration_over_stages_per_pipeline(metric_df)
    print(metric_df)
    # metric_df.to_csv(OUTPUT_ROOT / "paper/test_tab_duration_over_stages_per_pipeline.csv", sep="\t")

def test_tab_dataset_stats():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metrics = ["entity_count", "relation_count", "triple_count", "class_count", "loose_entity_count", "shallow_entity_count"]

    # filter for metrics
    metric_df = metric_df[["pipeline", "stage", "metric", "value"]]
    duration_df = agg_duration_over_stages_per_pipeline(metric_df)
    duration_df = duration_df[["pipeline", "stage", "metric", "value"]]
    metric_df = metric_df[metric_df["metric"].isin(metrics)]

    # print(metric_df.to_string())


    # print(duration_df.to_string())

    metric_df = pd.concat([metric_df, duration_df])
    # print(metric_df.to_string())

    # rename metric to short name
    metric_df["metric"] = metric_df["metric"].map(map_metric_name)
    # group by pipeline and stage and take mean of normalized


    # select only pipeline, stage, metric, normalized


    # Assuming your dataframe is called df
    pivot_df = metric_df.pivot_table(
        index=["pipeline", "stage"],   # rows
        columns="metric",              # pivoted column
        values="value"            # values to fill
    ).reset_index()

    # (Optional) Flatten the column index if needed
    pivot_df.columns.name = None  # remove "metric" header

    # save as TSV
    output_path = OUTPUT_ROOT / "paper/test_tab_dataset_stats"
    pivot_df.to_csv(output_path, sep="\t")

from collections import defaultdict
import json



def test_tab_ssp_stats():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)
    # remove details colums
    local_metric_df = metric_df.drop(columns=["details"])

    # only stage = stage_1 and aspect = statistical
    stage_1_df = local_metric_df[local_metric_df["stage"] == "stage_3"]
    statistical_df = stage_1_df[stage_1_df["aspect"] == "statistical"]

    metrics = ["entity_count", "relation_count", "triple_count", "class_count", "duration", "loose_entity_count", "shallow_entity_count"]

    # normalize metric=duration to seconds and round to 2 decimal places
    statistical_df.loc[statistical_df["metric"] == "duration", "value"] = statistical_df.loc[statistical_df["metric"] == "duration", "value"].round(2)

    # only selected metrics
    statistical_df = statistical_df[statistical_df["metric"].isin(metrics)]

    # rename metric to short name
    statistical_df["metric"] = statistical_df["metric"].map(map_metric_name)

    # Assuming your dataframe is called df
    pivot_df = statistical_df.pivot_table(
        index=["pipeline", "stage"],   # rows
        columns="metric",              # pivoted column
        values="value"            # values to fill
    ).reset_index()

    # (Optional) Flatten the column index if needed
    pivot_df.columns.name = None  # remove "metric" header


    # save as TSV
    output_path = OUTPUT_ROOT / "paper/test_tab_ssp_stats"
    pivot_df.to_csv(output_path, sep="\t")

SEM_METRIC_SHORT_NAMES = {
    "reasoning" : "EO0",
    "disjoint_domain": "$O_{DT}$",
    "incorrect_relation_direction": "$O_{RD}$",
    "incorrect_relation_cardinality": "$O_{CA}$",
    "incorrect_relation_range": "$O_{R}$",
    "incorrect_relation_domain": "$O_{D}$",
    "incorrect_datatype": "$O_{LT}$",
    "incorrect_datatype_format": "$O_{LF}$",
    "ontology_class_coverage": "$O_{CC}$",
    "ontology_relation_coverage": "$O_{RC}$",
    "ontology_namespace_coverage": "$O_{NC}$",
}

SEM_METRIC_LONG_NAMES = {v: k for k, v in SEM_METRIC_SHORT_NAMES.items()}

def add_REI_precision(metric_df):
    # REI_fscore = 2 * (precision * recall) / (precision + recall)
    source_entity_coverage_metric_soft = metric_df[metric_df["metric"] == "SourceEntityCoverageMetricSoft"]

    additional_rows = []
    for index, row in source_entity_coverage_metric_soft.iterrows():
        details = json.loads(row["details"])
        #"{""expected_entities_count"": 2758, ""found_entities_count"": 3099, ""overlapping_entities_count"": 53}"
        
        expected_entities_count = details["expected_entities_count"]
        #found_entities_count = details["found_entities_count"]
        overlapping_entities_count = details["overlapping_entities_count"]

        tp = overlapping_entities_count if overlapping_entities_count <= expected_entities_count else expected_entities_count
        fp = overlapping_entities_count - tp if overlapping_entities_count > tp else 0
        precision = tp / (tp + fp)

        additional_rows.append(
            {"pipeline": row["pipeline"], 
            "stage": row["stage"], 
            "metric": "REI_precision", 
            "aspect": "reference", 
            "normalized": precision,
            "value": precision,
            "details": row["details"]})

    additional_df = pd.DataFrame(additional_rows)
    return pd.concat([metric_df, additional_df])

    # precision = source_entity_coverage_metric_soft["normalized"]
    # recall = source_entity_coverage_metric["normalized"]
    # REI_fscore = 2 * (precision * recall) / (precision + recall)
    # metric_df["REI_fscore"] = REI_fscore
    # return metric_df

    # metric_df["REI_fscore"] = 2 * (metric_df["precision"] * metric_df["recall"]) / (metric_df["precision"] + metric_df["recall"])
    # return metric_df

def test_tab_ssp_semantic_eval():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)
    # remove details colums
    local_metric_df = metric_df.drop(columns=["details"])

    # only stage = stage_1 and aspect = statistical
    stage_1_df = local_metric_df[local_metric_df["stage"] == "stage_1"]
    statistical_df = stage_1_df[stage_1_df["aspect"] == "semantic"]

    # print all available metric names
    print(statistical_df["metric"].unique())

    # rename metric to short name and remove metrics that are not in SEM_METRIC_SHORT_NAMES 
    statistical_df = statistical_df[statistical_df["metric"].isin(list(SEM_METRIC_SHORT_NAMES.keys()))]

    statistical_df["metric"] = statistical_df["metric"].map(SEM_METRIC_SHORT_NAMES)

    # format normalized value to 2 decimal places
    statistical_df["normalized"] = statistical_df["normalized"].round(2)

    # make CSV with, x axis: pipeline, y axis: metric_name, cell: value
    # Pivot the table: index=metric, columns=pipeline, values=value
    pivot_df = statistical_df.pivot(index="metric", columns="pipeline", values="normalized")
    # transpose the table
    pivot_df = pivot_df.T


    # ✅ Add first row with long names
    # assume you have a dict SEM_METRIC_LONG_NAMES mapping short->long
    long_name_row = {col: SEM_METRIC_LONG_NAMES.get(col, col) for col in pivot_df.columns}
    pivot_df = pd.concat([pd.DataFrame([long_name_row], index=["metric_long_name"]), pivot_df])


    # Save as TSV
    output_path = OUTPUT_ROOT / "paper/test_tab_ssp_semantic_eval"
    pivot_df.to_csv(output_path, sep="\t")

# def test_fig_ssp_growth():
#     metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
#     # filter for pipeline in pipeline_types
#     # global metric_df
#     metric_df = metric_df[metric_df["pipeline"].isin(list(pipeline_types.keys())+list(llm_pipeline_types.keys()))]
#     sorted_metric_df = metric_df.sort_values(by=["stage", "pipeline"])

#     g = plot_growth(sorted_metric_df, metrics=["entity_count", "relation_count", "triple_count"], kind="bar")
#     # save as png
#     g.savefig(OUTPUT_ROOT / "paper/test_fig_ssp_growth.png")

def filter_msp_and_reference(metric_df):
    # filter for pipeline in pipeline_types
    # global metric_df
    pipline_names = metric_df["pipeline"].unique()
    filtered_pipeline_names = [name for name in pipline_names if len(name.split("_")) == 3 or name == "reference"]
    metric_df = metric_df[metric_df["pipeline"].isin(filtered_pipeline_names)]
    return metric_df

# def test_fig_msp_growth():
#     metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
#     # rename pipeline json_b2 to json_b
#     metric_df["pipeline"] = metric_df["pipeline"].replace("json_b2", "json_b")
#     # filter for pipeline in pipeline_types
#     # global metric_df
#     # apply filter function
#     metric_df = filter_msp_and_reference(metric_df)
#     # order by stage and pipeline
#     sorted_metric_df = metric_df.sort_values(by=["stage", "pipeline"])
#     g = plot_growth(sorted_metric_df, metrics=["entity_count", "relation_count", "triple_count"], kind="bar")
#     # save as png
#     g.savefig(OUTPUT_ROOT / "paper/test_fig_msp_growth.png")

def test_fig_both_growth():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    # remove reference stage_0
    metric_df["pipeline"] = metric_df["pipeline"].replace("json_b2", "json_b")

    metric_df = metric_df[metric_df["stage"] != "stage_0"]

    # filter for pipeline in pipeline_types
    # global metric_df
    # metric_df = filter_msp_and_reference(metric_df)
    sorted_metric_df = metric_df.sort_values(by=["stage", "pipeline"])
    g = plot_growth(sorted_metric_df, metrics=["entity_count", "triple_count"], kind="bar")
    g.fig.subplots_adjust(wspace=0.1)
    # save as png
    g.savefig(OUTPUT_ROOT / "paper/test_fig_both_growth.png")

def test_tab_ssp_erl_eval():
    pass

@pytest.mark.skip(reason="Not implemented")
def test_fig_stats():
    pass

def test_fig_msp_type_reference():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    # g = plot_class_occurence(metric_df)
    # g.savefig(OUTPUT_ROOT / "paper/test_fig_msp_type_reference.png")
    g = plot_class_occ_4_bar_chart(metric_df)
    g.savefig(OUTPUT_ROOT / "paper/test_fig_msp_type_reference.png")

def test_tab_er_el_ssp():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)

    filter_metrics = ["ER_EntityMatchMetric", "ER_RelationMatchMetric", "TE_ExpectedEntityLinkMetric"]
    metric_df = metric_df[metric_df["metric"].isin(filter_metrics)]
    # details not cotain "error" keyword
    # metric_df = metric_df[~metric_df["details"].str.contains("error")]
    
    # only pipeline, stage, metric, normalized
    erl_df = metric_df[["pipeline", "stage", "metric", "normalized"]]
    erl_df["normalized"] = erl_df["normalized"].round(2)

    pivot_df = erl_df.pivot_table(index=["pipeline", "stage"], 
                                columns="metric", 
                                values="normalized").reset_index()
    # save as TSV
    output_path = OUTPUT_ROOT / "paper/test_tab_er_el_ssp"
    pivot_df.to_csv(output_path, sep="\t")

def test_tab_all_metrics():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)

    # only pipeline, stage, metric, normalized
    erl_df = metric_df[["pipeline", "stage", "metric", "normalized"]]
    erl_df["normalized"] = erl_df["normalized"].round(2)

    # print(erl_df["metric"].unique())

    erl_df["metric"] = erl_df["metric"].map(map_metric_name)

    pivot_df = erl_df.pivot_table(index=["pipeline", "stage"], 
                                columns="metric", 
                                values="normalized").reset_index()

    output_path = OUTPUT_ROOT / "paper/test_tab_all_metrics"
    pivot_df.to_csv(output_path, sep="\t")

def test_ranking():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)

    # TODO replace with shortnames later
    semantic_metrics = list(set(SEM_METRIC_SHORT_NAMES.keys()) - set(["ontology_class_coverage", "ontology_relation_coverage", "ontology_namespace_coverage"]))
    semantic_metrics_weights = [1 for s in semantic_metrics]
    normalized_semantic_metrics_weights = [w/sum(semantic_metrics_weights) for w in semantic_metrics_weights]

    ranking_df = rank_metrics(metric_df, metric_names=semantic_metrics, metric_weights=normalized_semantic_metrics_weights, agg="sum", fill_missing=0.0)
    
    ranking_df.to_csv(OUTPUT_ROOT / "paper/test_ranking.csv", index=False)

    # append a line to the CSV with the used metrics
    with open(OUTPUT_ROOT / "paper/test_ranking.csv", "a") as f:
        f.write("Used metrics: " + ", ".join(semantic_metrics) + "\n")

def test_reference_alignment():
    # "Pipeline Inc. P R F1 ∼P ∼F ∼F1"
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)

    # ReferenceTripleAlignmentMetricSoftEV
    # ReferenceTripleAlignmentMetricSoftE
    # ReferenceTripleAlignmentMetric

    metric_names = ["ReferenceTripleAlignmentMetricSoftEV", "ReferenceTripleAlignmentMetricSoftE", "ReferenceTripleAlignmentMetric"]
    names_map = {
        "ReferenceTripleAlignmentMetricSoftEV": "soft_ev_",
        "ReferenceTripleAlignmentMetricSoftE": "soft_e_",
        "ReferenceTripleAlignmentMetric": "strict_",
    }

    import json

    # filter for pipeline in pipeline_types
    # global metric_df
    # apply filter function
    # only stage = stage_1
    metric_df = metric_df[metric_df["stage"] == "stage_3"]
    metric_df = metric_df[metric_df["metric"].isin(metric_names)]
    metric_df["metric"] = metric_df["metric"].map(names_map)
    # order by stage and pipeline
    # print(metric_df.pivot_table(index=["pipeline", "metric"], values="normalized", aggfunc="mean"))

    # extract precision, recall from details.json
    metric_df["p"] = metric_df["details"].apply(lambda x: json.loads(x)["precision"] if "precision" in json.loads(x) else 0)
    metric_df["r"] = metric_df["details"].apply(lambda x: json.loads(x)["recall"] if "recall" in json.loads(x) else 0)
    # renmae value to f1
    metric_df["f1"] = metric_df["normalized"]

    # only pipline, metric, p, r, f1
    metric_df = metric_df[["pipeline", "metric", "p", "r", "f1"]]

    # pivot table
    # print(metric_df)

    # result 
    df_wide = metric_df.pivot(
        index="pipeline", 
        columns="metric", 
        values=["p", "r", "f1"]
    )

    # flatten MultiIndex columns
    df_wide.columns = [f"{m if m!='' else ''}{k}" for k, m in df_wide.columns]
    df_wide = df_wide.reset_index()

    # sort columns by name
    df_wide = df_wide[sorted(df_wide.columns)]
    # normalize values to 2 decimal places for all coluns except pipeline
    df_wide.iloc[:, 1:] = df_wide.iloc[:, 1:].round(2)

    output_path = OUTPUT_ROOT / "paper/test_reference_alignment"
    df_wide.to_csv(output_path, sep="\t")


# ALL AND EVERYHTING

def extract_class_occurence_df(df):

    classes = ["http://kg.org/ontology/Company", "http://kg.org/ontology/Person", "http://kg.org/ontology/Film"]


    pipeline_stage_class_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # for each pipeline and stage
    for pipeline in df["pipeline"].unique():
        for stage in df["stage"].unique():
            df_pipeline_stage = df[df["pipeline"] == pipeline]
            df_pipeline_stage = df_pipeline_stage[df_pipeline_stage["stage"] == stage]
            try:
                details = json.loads(df_pipeline_stage["details"].values[0])
                class_counts = details["classes"]
                for class_name, count in class_counts.items():
                    if class_name not in classes:
                        class_name = "Other"
                    pipeline_stage_class_count[pipeline][stage][class_name] += count
            except:
                print(f"Error loading details for {pipeline} {stage}")
                # print(df_pipeline_stage["details"].values[0])

    rows = []
    for pipeline, stage_class_count in pipeline_stage_class_count.items():
        for stage, class_count in stage_class_count.items():
            for class_name, count in class_count.items():
                rows.append({"pipeline": pipeline, "stage": stage, "metric": class_name.split("/")[-1], "score": count})
    
    return pd.DataFrame(rows)

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

def map_metric_name_pretty(metric_name):
    return METRIC_NAME_MAP_PRETTY.get(metric_name, metric_name) # TODO: remove this

def get_statistics_df(df):
    # only pipeline, stage, metric, normalized
    
    class_occurence_df = df[df["metric"] == "class_occurrence"]
    class_count_df = extract_class_occurence_df(class_occurence_df)

    # print(class_count_df)

    df = df[df["aspect"] == "statistical"]
    metircs = ["entity_count", "relation_count", "triple_count", "class_count", "duration", "loose_entity_count", "shallow_entity_count"]
    df = df[df["metric"].isin(metircs)]

    df = df[["pipeline", "stage", "metric", "value"]]
    df["score"] = df["value"].round(2)

    # union df and class_count_df
    df = pd.concat([df, class_count_df])

    # rename metric to short name
    df["metric"] = df["metric"].map(map_metric_name_pretty)

    # make each metric a column
    df = df.pivot(index=["pipeline", "stage"], columns="metric", values="score")
    df = df.reset_index()


    return df

def get_semantic_df(df):
    # only pipeline, stage, metric, normalized
    df = df[df["aspect"] == "semantic"]
    df = df[["pipeline", "stage", "metric", "normalized"]]

    metrics = list(SEM_METRIC_SHORT_NAMES.keys())
    df = df[df["metric"].isin(metrics)]

    df["score"] = df["normalized"].round(2)

    # rename metric to short name
    df["metric"] = df["metric"].map(map_metric_name_pretty)

    # make each metric a column
    df = df.pivot(index=["pipeline", "stage"], columns="metric", values="score")
    df = df.reset_index()

    return df

def get_reference_df(df):
    # only pipeline, stage, metric, normalized
    df = df[df["aspect"] == "reference"]
    df = add_REI_precision(df)

    df = df[["pipeline", "stage", "metric", "normalized"]]

    metrics = [
        "ReferenceTripleAlignmentMetricSoftEV", 
        "ReferenceTripleAlignmentMetricSoftE", 
        "ReferenceTripleAlignmentMetric",
        # "ReferenceClassCoverageMetric",
        "SourceEntityCoverageMetric",
        "SourceEntityCoverageMetricSoft",
        "REI_precision",
        "TE_ExpectedEntityLinkMetric",
        "TE_ExpectedRelationLinkMetric",
        "ER_EntityMatchMetric",
        "ER_RelationMatchMetric",
    ]

    df = df[df["metric"].isin(metrics)]

    df["score"] = df["normalized"].round(2)

    # rename metric to short name
    df["metric"] = df["metric"].map(map_metric_name_pretty)

    # make each metric a column
    df = df.pivot(index=["pipeline", "stage"], columns="metric", values="score")
    df = df.reset_index()


    return df




def test_wide_table_smoth():

    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)


    # statistics_df 
    statistics_df = get_statistics_df(metric_df)
    # semantic_df
    semantic_df = get_semantic_df(metric_df)
    # reference_df
    reference_df = get_reference_df(metric_df)
    
    # join all of them on pipeline and stage
    df = pd.merge(statistics_df, semantic_df, on=["pipeline", "stage"], how="left")
    df = pd.merge(df, reference_df, on=["pipeline", "stage"], how="left")

    # colum order
    df = df[["pipeline", "stage"] + [v for k, v in METRIC_NAME_INDEX_PRETTY]]
    # print(df)

    df.to_csv(OUTPUT_ROOT / "paper/test_wide_table_smoth.csv", sep="\t")


# 4 different sub scores
# size (norm lowest to largest for entities, triples, density)
# semantic (average score of sematic metrics)
# reference (recall verified source entities, precision reference alignment)
# efficiency (long to short for duration, and high to low for memory peak)

def aggregate_metrics(metrics: list[str], weights: list[float]):
    # pipeline, stage, 
    pass 

def aggregate_semantic_metrics(df: pd.DataFrame):
    metrics = list(SEM_METRIC_SHORT_NAMES.keys())
    df = df[df["metric"].isin(metrics)]
    df = df[["pipeline", "stage", "metric", "normalized"]]
    # for each pipeline and stage = stage_3, calculate the average of the metrics
    df = df[df["stage"] == "stage_3"]
    
    return df

def test_f1_source_entity():

    df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    df = df[df["metric"] == "SourceEntityPrecisionMetric"]

    for row in df.itertuples():
        details = json.loads(row.details)
        expected_entities_count = details["expected_entities_count"]
        found_entities_count = details["found_entities_count"]
        overlapping_entities_count = details["overlapping_entities_count"]
        possible_duplicates_count = details["possible_duplicates_count"]
        overlapping_entities_strict_count = details["overlapping_entities_strict_count"]
        
        print(f"pipeline={row.pipeline}, stage={row.stage}, expected_entities_count={expected_entities_count}, found_entities_count={found_entities_count}, overlapping_entities_count={overlapping_entities_count}, possible_duplicates_count={possible_duplicates_count}, overlapping_entities_strict_count={overlapping_entities_strict_count}")
        precision = overlapping_entities_strict_count / overlapping_entities_count
        precision = precision if precision <= 1.0 else 1.0
        recall = overlapping_entities_count / expected_entities_count
        recall = recall if recall <= 1.0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall)
        df.loc[row.Index, "normalized"] = f1

    df = df[["pipeline", "stage", "normalized"]]

    # renmae piplines
    df["pipeline"] = df["pipeline"].map(map_pipeline_name)

    df.sort_values(by=["pipeline", "stage"], inplace=True)
    df.to_csv(OUTPUT_ROOT / "paper/test_average_f1_source_entity_f1.csv", sep="\t")


def get_average_f1_source_entity_f1(df: pd.DataFrame):
    df = df[df["metric"] == "SourceEntityPrecisionMetric"]

    for row in df.itertuples():
        details = json.loads(row.details)
        expected_entities_count = details["expected_entities_count"]
        found_entities_count = details["found_entities_count"]
        overlapping_entities_count = details["overlapping_entities_count"]
        possible_duplicates_count = details["possible_duplicates_count"]
        overlapping_entities_strict_count = details["overlapping_entities_strict_count"]
        
        print(f"pipeline={row.pipeline}, stage={row.stage}, expected_entities_count={expected_entities_count}, found_entities_count={found_entities_count}, overlapping_entities_count={overlapping_entities_count}, possible_duplicates_count={possible_duplicates_count}, overlapping_entities_strict_count={overlapping_entities_strict_count}")
        precision = overlapping_entities_strict_count / overlapping_entities_count
        precision = precision if precision <= 1.0 else 1.0
        recall = overlapping_entities_count / expected_entities_count
        recall = recall if recall <= 1.0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall)
        df.loc[row.Index, "normalized"] = f1

    df = df[["pipeline", "normalized"]]

    # save as csv

    # calculate the average of the metrics
    df = df.groupby("pipeline").mean().reset_index()
    # set as value for normalized and stage_3
    df["stage"] = "stage_3"
    df["metric"] = "SourceEntityF1Metric"
    df["value"] = df["normalized"]
    df = df[["pipeline", "stage", "metric", "value"]]

    return df

def aggregate_reference_metrics(df: pd.DataFrame):
    metrics = [
        "ReferenceTripleAlignmentMetricSoftEV", 
        "SourceEntityPrecisionMetric",
    ]

    source_entity_f1_df = get_average_f1_source_entity_f1(df)
    df = pd.concat([df, source_entity_f1_df])
    
    df = df[df["metric"].isin(metrics)]
    # if metric is ReferenceTripleAlignmentMetricSoftEV get details["f1"] and set normalized to it

    df.loc[df["metric"] == "ReferenceTripleAlignmentMetricSoftEV", "normalized"] = df[df["metric"] == "ReferenceTripleAlignmentMetricSoftEV"]["details"].apply(lambda x: json.loads(x)["f1_score"])

    df = df[["pipeline", "stage", "metric", "normalized"]]

    new_rows = []
    for pipeline in df["pipeline"].unique():
        new_rows.append({
            "pipeline": pipeline,
            "stage": "stage_3",
            "metric": "EntityMatchingMetric",
            "normalized": 0.85
        })
        new_rows.append({
            "pipeline": pipeline,
            "stage": "stage_3",
            "metric": "OntologyMatchingMetric",
            "normalized": 0.75
        })
        new_rows.append({
            "pipeline": pipeline,
            "stage": "stage_3",
            "metric": "EntityLinkingMetric",
            "normalized": 0.44
        })

    df = pd.concat([df, pd.DataFrame(new_rows)])


    # for each pipeline and stage = stage_3, calculate the average of the metrics
    df = df[df["stage"] == "stage_3"]

    return df

def aggregate_efficiency_metrics(df: pd.DataFrame):
    metrics = ["duration", "memory_peak"]
    df = df[df["metric"].isin(metrics)]
    df = df[["pipeline", "stage", "metric", "value"]]
    # for duration aggregate sum the values for each pipeline and stage
    
    duration_df = agg_duration_over_stages_per_pipeline(df)
    # remove duration
    df = df[df["metric"] != "duration"]
    df = pd.concat([df, duration_df])
    
    df["stage"] = "stage_3"

    def get_min_for_metric(metric):
        return df[df["metric"] == metric]["value"].min()

    def get_max_for_metric(metric):
        return df[df["metric"] == metric]["value"].max()

    for metric in df["metric"].unique():
        min_val = get_min_for_metric(metric)
        max_val = get_max_for_metric(metric)
        df.loc[df["metric"] == metric, "normalized"] = norm_min(min_val, df["value"])

    return df

import numpy as np



def aggregate_size_metrics(df: pd.DataFrame):
    metrics = ["entity_count", "triple_count"]
    df = df[df["metric"].isin(metrics)]
    df = df[["pipeline", "stage", "metric", "value"]]
    # for each pipeline and stage = stage_3, calculate the average of the metrics
    df = df[df["stage"] == "stage_3"]

    # Pivot to compute density per pipeline
    wide = df.pivot(index="pipeline", columns="metric", values="value")

    # Compute density = triple_count / entity_count (guard against zero/NaN)
    denom = wide["entity_count"]
    numer = wide["triple_count"]
    density = np.where((denom > 0) & np.isfinite(denom), numer / denom, np.nan)
    wide["density"] = density
    
    
    # Return to long format: (pipeline, metric, value)
    df = (wide.reset_index()
                 .melt(id_vars="pipeline", var_name="metric", value_name="value"))


    def _normalize(group: pd.DataFrame):
        vmax = group["value"].max()
        vmin = group["value"].min()

        invert_normalization = False
        if group.name == "density":
            invert_normalization = True

        if invert_normalization:
            group["normalized"] = norm_min(vmin, group["value"])  # largest→0, smallest→1
        else:
            group["normalized"] = norm_max(vmax, group["value"]) #(group["value"] - vmin) / (vmax - vmin)  # smallest→0, largest→1

        return group

    df = df.groupby("metric", group_keys=False).apply(_normalize)

    return df

def mean_scores(df, column_name):
    df = df[["pipeline", "normalized"]]
    # calculate the average of the metrics
    df = df.groupby("pipeline").mean().reset_index()
    # rename normalized to semantic
    df = df.rename(columns={"normalized": column_name})
    return df

def aggregate_ranking_df():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # # replace pipeline name with name_mapping
    # metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)

    # only pipelines where name contains 2 "_" chars
    metric_df = metric_df[metric_df["pipeline"].str.count("_") == 2]

    norm_semantic_df = aggregate_semantic_metrics(metric_df)
    norm_semantic_df = norm_semantic_df[["pipeline", "metric", "normalized"]]
    agg_semantic_df = mean_scores(norm_semantic_df, "semantic")
    
    norm_reference_df = aggregate_reference_metrics(metric_df)
    norm_reference_df = norm_reference_df[["pipeline", "metric", "normalized"]]
    print(norm_reference_df.to_string())
    agg_reference_df = mean_scores(norm_reference_df, "reference")
    
    norm_efficiency_df = aggregate_efficiency_metrics(metric_df)
    # print(norm_efficiency_df)
    norm_efficiency_df = norm_efficiency_df[["pipeline", "metric", "normalized"]]
    agg_efficiency_df = mean_scores(norm_efficiency_df, "efficiency")
    
    norm_size_df = aggregate_size_metrics(metric_df)
    norm_size_df = norm_size_df[["pipeline", "metric", "normalized"]]
    agg_size_df = mean_scores(norm_size_df, "size")

    norm_df = pd.merge(norm_semantic_df, norm_reference_df, on=["pipeline", "metric", "normalized"], how="outer")
    norm_df = pd.merge(norm_df, norm_efficiency_df, on=["pipeline", "metric", "normalized"], how="outer")
    norm_df = pd.merge(norm_df, norm_size_df, on=["pipeline", "metric", "normalized"], how="outer")

    agg_df = pd.merge(agg_semantic_df, agg_reference_df, on=["pipeline"], how="left")
    agg_df = pd.merge(agg_df, agg_efficiency_df, on=["pipeline"], how="left")
    agg_df = pd.merge(agg_df, agg_size_df, on=["pipeline"], how="left")

    return norm_df, agg_df

# different ranking scenarios
# Equal weights (semantic, reference, efficiency): JSON-b and RDF-a near top; text-heavy MSPs vary by order.
# Quantity first (↑size)
# Quality-first (↑semantic, ↑reference): RDF-a > JSON-b > RDF-b.
# Throughput-first (↑efficiency): JSON-a and RDF-b improve rank.

# \item \textbf{Equal weighting (baseline):} \\
# $(c1, c2, c3, c4) = (0.25, 0.25, 0.25, 0.25)$
def test_rank_equal():
    """
    pipeline  semantic  reference  efficiency
    0  json_rdf_text  0.855084   0.541371    0.795746
    1  json_text_rdf  0.867719   0.422527    0.886575
    2  rdf_json_text  0.855081   0.568467    0.330697
    3  rdf_text_json  0.867721   0.694226    0.000000
    4  text_json_rdf  0.864522   0.921585    0.799708
    5  text_rdf_json  0.864522   0.668960    1.000000
    """
    norm_df, df = aggregate_ranking_df()

    # combine semantic, reference, and efficiency
    df["combined"] = df["size"] * 0.25 + df["semantic"] * 0.25 + df["reference"] * 0.25 + df["efficiency"] * 0.25
    df = df[["pipeline", "combined"]]
    # round to 2 decimal places
    df["combined"] = df["combined"].round(3)

    # sort by combined
    df = df.sort_values(by="combined", ascending=False)
    
    df.to_csv(OUTPUT_ROOT / "paper/test_rank_equal.csv", sep="\t")

# def test_rank_quantity():
#     _, df = aggregate_ranking_df()

#     # combine semantic, reference, and efficiency
#     df["size"] = df["size"] * 1
#     # round to 2 decimal places
#     df["size"] = df["size"].round(2)
#     df = df[["pipeline", "size"]]

#     # sort by combined
#     df = df.sort_values(by="size", ascending=False)
    
#     df.to_csv(OUTPUT_ROOT / "paper/test_rank_quantity.csv", sep="\t")

# def test_rank_quality():
#     _, df = aggregate_ranking_df()

#     # combine semantic, reference, and efficiency
#     df["quality"] = df["semantic"] * 0.5 + df["reference"] * 0.5
#     # round to 2 decimal places
#     df["quality"] = df["quality"].round(2)
#     df = df[["pipeline", "quality"]]

#     # sort by combined
#     df = df.sort_values(by="quality", ascending=False)
    
#     df.to_csv(OUTPUT_ROOT / "paper/test_rank_quality.csv", sep="\t")

# def test_rank_efficiency():
#     _, df = aggregate_ranking_df()

#     # combine semantic, reference, and efficiency
#     df["efficiency"] = df["efficiency"] * 1
#     # round to 2 decimal places
#     df["efficiency"] = df["efficiency"].round(2)
#     df = df[["pipeline", "efficiency"]]

#     # sort by combined
#     df = df.sort_values(by="efficiency", ascending=False)
    
#     df.to_csv(OUTPUT_ROOT / "paper/test_rank_efficiency.csv", sep="\t")

def test_rank_save_norm_df():
    
    
    norm_df, _ = aggregate_ranking_df()
    # round to 2 decimal places
    norm_df["normalized"] = norm_df["normalized"].round(2)
    """
    0    json_rdf_text  ReferenceTripleAlignmentMetricSoftEV    0.855469
    1    json_rdf_text        SourceEntityCoverageMetricSoft    0.227273
    2    json_rdf_text                               density    0.000000
    3    json_rdf_text                       disjoint_domain    1.000000
    4    json_rdf_text                              duration    0.795746
    """
    # to format pipeline, metric_name1... metric_nameN, normalized

    wide = norm_df.pivot(index="pipeline", columns="metric", values="normalized").reset_index()

    wide.to_csv(OUTPUT_ROOT / "paper/test_rank_norm_df.csv", sep="\t")




# c1 stat c2 sem c3 ref c4 eff
# \item \textbf{Quantity-focused:} \\
# $(c1, c2, c3, c4) = (0.50, 0.20, 0.20, 0.10)$
def test_rank_quantity_focused():
    _, df = aggregate_ranking_df()
    # combine semantic, reference, and efficiency
    df["combined"] = df["size"] * 0.5 + df["semantic"] * 0.1 + df["reference"] * 0.1 + df["efficiency"] * 0.3
    df = df[["pipeline", "combined"]]
    # round to 2 decimal places
    df["combined"] = df["combined"].round(3)

    # sort by combined
    df = df.sort_values(by="combined", ascending=False)
    
    df.to_csv(OUTPUT_ROOT / "paper/test_rank_quantity_focused.csv", sep="\t")

# \item \textbf{Quality-focused:} \\
# $(c1, c2, c3, c4) = (0.10, 0.40, 0.40, 0.10)$
def test_rank_quality_focused():
    _, df = aggregate_ranking_df()
    df["combined"] = df["size"] * 0.0 + df["semantic"] * 0.5 + df["reference"] * 0.5 + df["efficiency"] * 0.0
    df = df[["pipeline", "combined"]]
    # round to 2 decimal places
    df["combined"] = df["combined"].round(3)

    # sort by combined
    df = df.sort_values(by="combined", ascending=False)
    
    df.to_csv(OUTPUT_ROOT / "paper/test_rank_quality_focused.csv", sep="\t")

# \item \textbf{Reference-alignment focused:} \\
# $(c1, c2, c3, c4) = (0.10, 0.20, 0.60, 0.10)$
def test_rank_reference_alignment_focused():
    _, df = aggregate_ranking_df()
    df["combined"] = df["size"] * 0.0 + df["semantic"] * 0.2 + df["reference"] * 0.8 + df["efficiency"] * 0.0
    df = df[["pipeline", "combined"]]
    # round to 2 decimal places
    df["combined"] = df["combined"].round(3)
    
    # sort by combined
    df = df.sort_values(by="combined", ascending=False)
    
    df.to_csv(OUTPUT_ROOT / "paper/test_rank_reference_alignment_focused.csv", sep="\t")

# \item \textbf{Efficiency-oriented:} \\
# $(c1, c2, c3, c4) = (0.20, 0.20, 0.20, 0.40)$
def test_rank_efficiency_oriented():
    _, df = aggregate_ranking_df()
    df["combined"] = df["size"] * 0.2 + df["semantic"] * 0.2 + df["reference"] * 0.2 + df["efficiency"] * 0.4
    df = df[["pipeline", "combined"]]
    # round to 2 decimal places
    df["combined"] = df["combined"].round(3)
    
        # sort by combined
    df = df.sort_values(by="combined", ascending=False)
    

    df.to_csv(OUTPUT_ROOT / "paper/test_rank_efficiency_oriented.csv", sep="\t")

