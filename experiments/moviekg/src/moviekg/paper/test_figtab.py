import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

from moviekg.config import OUTPUT_DIR, DATASET_SELECT
from moviekg.paper.helpers.agggregate import agg_duration_over_stages_per_pipeline
from moviekg.paper.helpers.getter import get_pipeline_stage_metric_dict, TABLE_DISPLAY_NAMES, apply_selected_updates
from moviekg.paper.helpers.helpers import load_metrics_from_file, plot_growth, plot_class_occ_4_bar_chart
from moviekg.paper.helpers.ranking import _rank_and_save2csv
from moviekg.paper.config import (
    name_mapping, METRIC_NAME_MAP, SEM_METRIC_SHORT_NAMES, 
    METRIC_NAME_INDEX_PRETTY, METRIC_NAME_MAP_PRETTY, SEM_METRIC_LONG_NAMES
)


# === Preamble ===
if not OUTPUT_DIR:
    raise ValueError("OUTPUT_DIR is not set")
if not DATASET_SELECT:
    raise ValueError("DATASET_SELECT is not set")

OUTPUT_ROOT = Path(OUTPUT_DIR) / DATASET_SELECT
(OUTPUT_ROOT / "paper").mkdir(parents=True, exist_ok=True)

PIPLEINE_NAME_MAP = {
    "json_rdf_text": "JRT",
    "json_text_rdf": "JTR",
    "rdf_json_text": "RJT",
    "rdf_text_json": "RTJ",
    "text_json_rdf": "TJR",
    "text_rdf_json": "TRJ",
    "json_a": "J_A",
    "json_b": "J_B",
    "json_c": "J_C",
    "json_llm_mapping_v1": "J_C",
    "json_baseA": "J_baseA",
    "rdf_a": "R_A",
    "rdf_b": "R_B",
    "rdf_c": "R_C",
    "rdf_llm_schema_align_v1": "R_C",
    "text_a": "T_A",
    "text_b": "T_B",
    "text_c": "T_C",
    "text_llm_triple_extract_v1": "T_C",
    }

def map_pipeline_name_pretty(pipeline_name):
    return PIPLEINE_NAME_MAP.get(pipeline_name, pipeline_name)

# === Helper Functions ===

def map_pipeline_name(pipeline_name):
    return name_mapping.get(pipeline_name, pipeline_name)


def map_metric_name(metric_name):
    return METRIC_NAME_MAP.get(metric_name, metric_name)


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

def extract_class_occurence_df(df):

    classes = ["http://kg.org/ontology/Film", "http://kg.org/ontology/Person", "http://kg.org/ontology/Company"]


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
    df[["pipeline"]] = df[["pipeline"]].map(map_pipeline_name)

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
    # TODO metric names and selection
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

# === Tests ===

def test_wide_table_smoth():
    """
    Stores all metrics in a wide table format.
    """

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


def test_table_with_statistic_metrics():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metrics = ["entity_count", "relation_count", "triple_count", "class_count", "loose_entity_count", "shallow_entity_count"]

    # filter for metrics
    metric_df = metric_df[["pipeline", "stage", "metric", "value"]]
    duration_df = agg_duration_over_stages_per_pipeline(metric_df)
    duration_df = duration_df[["pipeline", "stage", "metric", "value"]]
    metric_df = metric_df[metric_df["metric"].isin(metrics)]

    metric_df = pd.concat([metric_df, duration_df])

    metric_df["metric"] = metric_df["metric"].map(map_metric_name)
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name_pretty)
    # only stage = stage_3
    metric_df = metric_df[metric_df["stage"] == "stage_3"]

    # Assuming your dataframe is called df
    pivot_df = metric_df.pivot_table(
        index=["pipeline", "stage"],   # rows
        columns="metric",              # pivoted column
        values="value"            # values to fill
    ).reset_index()

    # (Optional) Flatten the column index if needed
    pivot_df.columns.name = None  # remove "metric" header

    # column selection and order Pipeline FC EC RC TC Time
    pivot_df = pivot_df[["pipeline", "FC", "EC", "RC", "TC", "SEC", "Time"]]
    # save as TSV
    output_path = OUTPUT_ROOT / "paper/test_tab_2_statistic_metrics.csv"
    pivot_df.to_csv(output_path, sep="\t")


def test_table_with_semantic_metrics():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name_pretty)
    # remove details colums
    local_metric_df = metric_df.drop(columns=["details"])

    # only stage = stage_1 and aspect = statistical
    stage_3_df = local_metric_df[local_metric_df["stage"] == "stage_3"]
    statistical_df = stage_3_df[stage_3_df["aspect"] == "semantic"]
    # statistical_df["pipeline"] = statistical_df["pipeline"].map(map_pipeline_name_pretty)

    # print all available metric names
    print(statistical_df["metric"].unique())

    # rename metric to short name and remove metrics that are not in SEM_METRIC_SHORT_NAMES 
    statistical_df = statistical_df[statistical_df["metric"].isin(list(SEM_METRIC_SHORT_NAMES.keys()))]
    statistical_df["metric"] = statistical_df["metric"].map(SEM_METRIC_SHORT_NAMES)

    # format normalized value to 2 decimal places
    statistical_df["normalized"] = statistical_df["normalized"].round(3)

    # only stage = stage_3
    statistical_df = statistical_df[statistical_df["stage"] == "stage_3"]

    # make CSV with, x axis: pipeline, y axis: metric_name, cell: value
    # Pivot the table: index=metric, columns=pipeline, values=value
    pivot_df = statistical_df.pivot(index="metric", columns="pipeline", values="normalized")
    # transpose the table
    pivot_df = pivot_df.T

    # assume you have a dict SEM_METRIC_LONG_NAMES mapping short->long
    long_name_row = {col: SEM_METRIC_LONG_NAMES.get(col, col) for col in pivot_df.columns}
    pivot_df = pd.concat([pd.DataFrame([long_name_row], index=["metric_long_name"]), pivot_df])

    # column selection and order pipeline 𝑂𝐷𝑇 𝑂𝐷 𝑂𝑅 𝑂𝑅𝐷 𝑂𝐿𝑇 𝑂𝐿𝐹 𝑂𝐴𝑣𝑔

    output_path = OUTPUT_ROOT / "paper/test_tab_3_ssp_semantic_eval.csv"
    pivot_df.to_csv(output_path, sep="\t")

def test_table_with_matching_metrics():
    from moviekg.paper.helpers.getter import TABLE_DISPLAY_NAMES, get_pipeline_stage_metric_dict, ref_entity_matching_f1, ref_relation_matching_f1, ref_json_entity_matching_f1

    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name_pretty)
    metrics = [metric for metric in list(TABLE_DISPLAY_NAMES.keys()) if metric in [ref_entity_matching_f1.__name__, ref_relation_matching_f1.__name__, ref_json_entity_matching_f1.__name__]]

    metric_dict = get_pipeline_stage_metric_dict(metric_df, metrics)
    
    df_rows = []
    for pipeline, stage_dict in metric_dict.items():
        for stage, metric_dict in stage_dict.items():
            rdf_em_f1 = metric_dict.get(ref_entity_matching_f1.__name__, -1)
            json_em_f1 = metric_dict.get(ref_json_entity_matching_f1.__name__, -1)
            em_f1 = -1
            if rdf_em_f1 != -1:
                em_f1 = rdf_em_f1
            elif json_em_f1 != -1:
                em_f1 = json_em_f1

            rdf_rm_f1 = metric_dict.get(ref_relation_matching_f1.__name__, -1)
            json_el_r = -1 # metric_dict.get(ref.__name__, -1)
            rm_f1 = -1
            if rdf_rm_f1 != -1:
                rm_f1 = rdf_rm_f1
            elif json_el_r != -1:
                rm_f1 = json_el_r

            df_rows.append({"pipeline": pipeline, "stage": stage, "EM_f1": em_f1, "RM_f1": rm_f1})

    # remove -1 rows
    df_rows = [row for row in df_rows if row["EM_f1"] != -1 and row["RM_f1"] != -1]

    df = pd.DataFrame(df_rows)
    # df = df.pivot(index=["pipeline", "stage"], columns="metric", values="value")
    # df = df.reset_index()
    output_path = OUTPUT_ROOT / "paper/test_tab_4_matching_metrics.csv"
    df.to_csv(output_path, sep="\t")

def test_table_with_matching_metrics_pr():
    from moviekg.paper.helpers.getter import (
        TABLE_DISPLAY_NAMES, get_pipeline_stage_metric_dict, 
        ref_entity_matching_f1, ref_entity_matching_p, ref_entity_matching_r,
        ref_relation_matching_f1, ref_relation_matching_p, ref_relation_matching_r, 
        ref_json_entity_matching_f1, ref_json_entity_matching_p, ref_json_entity_matching_r
    )

    

    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name_pretty)
    metrics = [
        ref_entity_matching_p.__name__, ref_entity_matching_r.__name__,
        ref_relation_matching_p.__name__, ref_relation_matching_r.__name__, 
        ref_json_entity_matching_p.__name__, ref_json_entity_matching_r.__name__
    ]

    psmd = get_pipeline_stage_metric_dict(metric_df, metrics)
    
    df_rows = []
    for pipeline, stage_dict in psmd.items():    
        for stage, metric_dict in stage_dict.items():
            rdf_em_p = metric_dict.get(ref_entity_matching_p.__name__, -1)
            rdf_em_r = metric_dict.get(ref_entity_matching_r.__name__, -1)
            json_em_p = metric_dict.get(ref_json_entity_matching_p.__name__, -1)
            json_em_r = metric_dict.get(ref_json_entity_matching_r.__name__, -1)
            em_p = -1
            em_r = -1
            if rdf_em_p != -1:
                em_p = rdf_em_p
                em_r = rdf_em_r
            elif json_em_p != -1:
                em_p = json_em_p
                em_r = json_em_r

            # print(json.dumps(metric_dict, indent=4))
            # print("--------------------------------")

            rdf_rm_p = metric_dict.get(ref_relation_matching_p.__name__, -1)
            rdf_rm_r = metric_dict.get(ref_relation_matching_r.__name__, -1)
            json_rm_p = metric_dict.get(ref_relation_matching_p.__name__, -1)
            json_rm_r = metric_dict.get(ref_relation_matching_r.__name__, -1)

            rm_p = -1
            rm_r = -1
            if rdf_rm_p != -1:
                rm_p = rdf_rm_p
                rm_r = rdf_rm_r
            elif json_rm_p != -1:
                rm_p = json_rm_p
                rm_r = json_rm_r

            df_rows.append({"pipeline": pipeline, "stage": stage, "EM_p": em_p, "EM_r": em_r, "RM_p": rm_p, "RM_r": rm_r})

    # remove -1 rows
    df_rows = [row for row in df_rows if row["EM_p"] != -1 and row["EM_r"] != -1 and row["RM_p"] != -1 and row["RM_r"] != -1]

    df = pd.DataFrame(df_rows)
    # df = df.pivot(index=["pipeline", "stage"], columns="metric", values="value")
    # df = df.reset_index()
    output_path = OUTPUT_ROOT / "paper/test_tab_4_matching_metrics_pr.csv"
    df.to_csv(output_path, sep="\t")

def test_table_with_linking_metrics():
    from moviekg.paper.helpers.getter import TABLE_DISPLAY_NAMES, get_pipeline_stage_metric_dict, ref_entity_linking_r, ref_json_entity_linking_r

    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name_pretty)
    metrics = [metric for metric in list(TABLE_DISPLAY_NAMES.keys()) if metric in [ref_entity_linking_r.__name__, ref_json_entity_linking_r.__name__]]

    metric_dict = get_pipeline_stage_metric_dict(metric_df, metrics)
    
    df_rows = []
    for pipeline, stage_dict in metric_dict.items():
        for stage, metric_dict in stage_dict.items():
            rdf_el_r = metric_dict.get(ref_entity_linking_r.__name__, -1)
            json_el_r = metric_dict.get(ref_json_entity_linking_r.__name__, -1)
            el_r = -1
            if rdf_el_r != -1:
                el_r = rdf_el_r
            elif json_el_r != -1:
                el_r = json_el_r

            df_rows.append({"pipeline": pipeline, "stage": stage, "EL_r": el_r})

    # remove -1 rows
    df_rows = [row for row in df_rows if row["EL_r"] != -1]

    df = pd.DataFrame(df_rows)
    # df = df.pivot(index=["pipeline", "stage"], columns="metric", values="value")
    # df = df.reset_index()
    output_path = OUTPUT_ROOT / "paper/test_tab_5_linking_metrics.csv"
    df.to_csv(output_path, sep="\t")


def test_table_6():
    """
    External KG R @inc (film)
    EC (no Seed) REI @inc (film)
    Pipeline | f1@1 f1@2 f1@3 p@3 | f1@1 f@2 f@3
    """
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name_pretty)
    from moviekg.paper.helpers.getter import (
        get_pipeline_stage_metric_dict, ref_kg_f1, ref_kg_p, ref_kg_r, ref_source_entity_f1, ref_source_entity_p, ref_source_entity_r
    )

    metrics = [
        ref_kg_f1.__name__, ref_kg_p.__name__, ref_kg_r.__name__, ref_source_entity_f1.__name__, ref_source_entity_p.__name__, ref_source_entity_r.__name__
    ]

    psmd = get_pipeline_stage_metric_dict(metric_df, metrics)
    # import json
    # json.dump(psmd, open(OUTPUT_ROOT / "paper/test_tab_6_metrics.json", "w"), indent=4)

    rows = []

    round_to = 2

    for pipeline, stage_dict in psmd.items():
        if pipeline in ["reference", "seed"]:
            continue
        kg_p = [0, 0, 0]
        kg_r = [0, 0, 0]
        se_p = [0, 0, 0]
        se_r = [0, 0, 0]


        for stage, metric_dict in stage_dict.items():
            kg_p[int(stage.split("_")[1]) - 1] = round(metric_dict.get(ref_kg_p.__name__, -1), round_to)
            kg_r[int(stage.split("_")[1]) - 1] = round(metric_dict.get(ref_kg_r.__name__, -1), round_to)
            se_p[int(stage.split("_")[1]) - 1] = round(metric_dict.get(ref_source_entity_p.__name__, -1), round_to)
            se_r[int(stage.split("_")[1]) - 1] = round(metric_dict.get(ref_source_entity_r.__name__, -1), round_to)
        
        rows.append({
            "pipeline": pipeline, 
            "kg_p@1": kg_p[0], "kg_r@1": kg_r[0], "kg_p@2": kg_p[1], "kg_r@2": kg_r[1], "kg_p@3": kg_p[2], "kg_r@3": kg_r[2],
            "se_p@1": se_p[0], "se_r@1": se_r[0], "se_p@2": se_p[1], "se_r@2": se_r[1], "se_p@3": se_p[2], "se_r@3": se_r[2]})

    df = pd.DataFrame(rows)
    output_path = OUTPUT_ROOT / "paper/test_tab_6_reference_alignment.csv"
    df.to_csv(output_path, sep="\t")

def test_table_with_reference_overlap_metrics():
    # "Pipeline Inc. P R F1 ∼P ∼F ∼F1"
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")

    # replace pipeline name with name_mapping
    metric_df["pipeline"] = metric_df["pipeline"].map(map_pipeline_name)

    metric_names = ["ReferenceTripleAlignmentMetricSoftEV", "ReferenceTripleAlignmentMetricSoftE", "ReferenceTripleAlignmentMetric"]
    names_map = {
        "ReferenceTripleAlignmentMetricSoftEV": "soft_ev_",
        "ReferenceTripleAlignmentMetricSoftE": "soft_e_",
        "ReferenceTripleAlignmentMetric": "strict_",
    }

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

def test_figure_with_kg_growth():
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


def test_figure_with_entity_class_occurence():
    metric_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
    g = plot_class_occ_4_bar_chart(metric_df)
    g.savefig(OUTPUT_ROOT / "paper/test_fig_msp_type_reference.png")


# Preset weight configs (kept exactly as used in your original code)
PRESETS = {
    "equal": {
        "size": 0.25, "semantic": 0.25, "reference": 0.25, "efficiency": 0.25
    },
    # Quantity-focused (your code used 0.5, 0.1, 0.1, 0.3)
    "quantity_focused": {
        "size": 0.5, "semantic": 0.1, "reference": 0.1, "efficiency": 0.3
    },
    # Quality-focused (your code used 0.0, 0.5, 0.5, 0.0)
    "quality_focused": {
        "size": 0.0, "semantic": 0.5, "reference": 0.5, "efficiency": 0.0
    },
    # Reference-alignment focused (your code used 0.0, 0.2, 0.8, 0.0)
    "reference_alignment_focused": {
        "size": 0.0, "semantic": 0.2, "reference": 0.8, "efficiency": 0.0
    },
    "efficiency_oriented": {
        "size": 0.2, "semantic": 0.2, "reference": 0.2, "efficiency": 0.4
    },
}

psmd_df = load_metrics_from_file(OUTPUT_ROOT / "all_metrics.csv")
psmd = get_pipeline_stage_metric_dict(psmd_df, TABLE_DISPLAY_NAMES.keys())
psmd = apply_selected_updates(psmd)

# TODO cleanup
# norm_df, agg_df = aggregate_ranking_df()
# def test_rank_save_norm_df():
#     norm_df["normalized"] = norm_df["normalized"].round(2)
#     # to format pipeline, metric_name1... metric_nameN, normalized
#     wide = norm_df.pivot(index="pipeline", columns="metric", values="normalized").reset_index()
#     wide.to_csv(OUTPUT_ROOT / "paper/test_rank_norm_df.csv", sep="\t")

# c1 size, c2 sem, c3 ref, c4 eff
def test_rank_equal():
    # _rank_and_save(PRESETS["equal"], "test_rank_equal", agg_df)
    _rank_and_save2csv(PRESETS["equal"], "test_rank_equal", psmd)

def test_rank_quantity_focused():
    #_rank_and_save(PRESETS["quantity_focused"], "test_rank_quantity_focused", agg_df)
    _rank_and_save2csv(PRESETS["quantity_focused"], "test_rank_quantity_focused", psmd)

def test_rank_quality_focused():
    #_rank_and_save(PRESETS["quality_focused"], "test_rank_quality_focused", agg_df)
    _rank_and_save2csv(PRESETS["quality_focused"], "test_rank_quality_focused", psmd)

def test_rank_reference_alignment_focused():
    #_rank_and_save(PRESETS["reference_alignment_focused"], "test_rank_reference_alignment_focused", agg_df)
    _rank_and_save2csv(PRESETS["reference_alignment_focused"], "test_rank_reference_alignment_focused", psmd)

def test_rank_efficiency_oriented():
    #_rank_and_save(PRESETS["efficiency_oriented"], "test_rank_efficiency_oriented", agg_df)
    _rank_and_save2csv(PRESETS["efficiency_oriented"], "test_rank_efficiency_oriented", psmd)

def test_full_ranking_table():
    """
    for each rank table read it and then concatenate them into one table joining on the index
    for example:
        test_rank_equal.csv:
            pipeline combined
            0 json_rdf_text 0.855084
            1 json_text_rdf 0.867719
            2 rdf_json_text 0.855081
            3 rdf_text_json 0.867721
            4 text_json_rdf 0.864522
            5 text_rdf_json 0.864522
        test_rank_quantity_focused.csv:
            pipeline combined
            0 rdf_json_text 0.950847
            1 text_rdf_json 0.940847
            2 json_text_rdf 0.93847
            3 rdf_text_json 0.920847
            4 json_rdf_text 0.910847
            5 text_json_rdf 0.900847

    the result should be:
    pipeline combined
    0 json_rdf_text 0.855084 rdf_json_text_0.950847
    1 json_text_rdf 0.867719 text_rdf_json_0.940847
    2 rdf_json_text 0.855081 json_text_rdf_0.93847
    3 rdf_text_json 0.867721 rdf_text_json_0.920847
    4 text_json_rdf 0.864522 json_rdf_text_0.910847
    5 text_rdf_json 0.864522 text_json_rdf_0.900847

    rename the "combined" column for each to the name of the file
    """

    ranking_files = [
        "test_rank_equal.csv",
        "test_rank_quantity_focused.csv",
        "test_rank_quality_focused.csv",
        "test_rank_reference_alignment_focused.csv",
        "test_rank_efficiency_oriented.csv"
    ]


    ranking_files = [OUTPUT_ROOT / "paper" / file for file in ranking_files]

    # Base frame with fixed ranks 0..5 (top to bottom)
    result = pd.DataFrame({"rank": range(15)})
    # result = pd.DataFrame()

    for file in ranking_files:
        name = Path(file).stem  # e.g., "test_rank_equal"
        df = pd.read_csv(file, sep="\t")
        # Ensure we have at least 6 rows; if more, keep top-6; if fewer, allow NaNs
        # df = df.head(6).reset_index(drop=True)

        # pipeline name != reference and reset index
        df = df[df["pipeline"] != "reference"]
        df["pipeline"] = df["pipeline"].map(PIPLEINE_NAME_MAP)
        df = df.reset_index(drop=True)
    

        # Build two columns for this file: pipeline + score
        sub = pd.DataFrame({
            "rank": df.index,
            f"{name.split(".")[0]}_pipe": df["pipeline"],
            f"{name.split(".")[0]}_score": df["combined"]
        })

        # Join on rank to keep rows aligned 0..5
        result = result.merge(sub, on="rank", how="left")

    # Make 'rank' the index if you prefer, or keep as a column
    result = result.set_index("rank")

    result.to_csv(OUTPUT_ROOT / "paper/test_tab_7_full_ranking_table.csv", sep="\t")
