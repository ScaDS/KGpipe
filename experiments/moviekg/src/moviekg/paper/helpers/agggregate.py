import pandas as pd
import json
import numpy as np
from moviekg.paper.config import SEM_METRIC_SHORT_NAMES
from moviekg.paper.helpers.helpers import load_metrics_from_file
from moviekg.config import OUTPUT_ROOT

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

def get_average_f1_source_entity_f1(df: pd.DataFrame):
    df = df[df["metric"] == "SourceEntityPrecisionMetric"]

    for row in df.itertuples():
        details = json.loads(row.details)
        expected_entities_count = details["expected_entities_count"]
        found_entities_count = details["found_entities_count"]
        overlapping_entities_count = details["overlapping_entities_count"]
        possible_duplicates_count = details["possible_duplicates_count"]
        overlapping_entities_strict_count = details["overlapping_entities_strict_count"]
        
        # print(f"pipeline={row.pipeline}, stage={row.stage}, expected_entities_count={expected_entities_count}, found_entities_count={found_entities_count}, overlapping_entities_count={overlapping_entities_count}, possible_duplicates_count={possible_duplicates_count}, overlapping_entities_strict_count={overlapping_entities_strict_count}")
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


def aggregate_semantic_metrics(df: pd.DataFrame):
    metrics = list(SEM_METRIC_SHORT_NAMES.keys())
    df = df[df["metric"].isin(metrics)]
    df = df[["pipeline", "stage", "metric", "normalized"]]
    # for each pipeline and stage = stage_3, calculate the average of the metrics
    df = df[df["stage"] == "stage_3"]
    
    return df

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
    # metric_df = metric_df[metric_df["pipeline"].str.count("_") == 2] TODO

    norm_semantic_df = aggregate_semantic_metrics(metric_df)
    norm_semantic_df = norm_semantic_df[["pipeline", "metric", "normalized"]]
    agg_semantic_df = mean_scores(norm_semantic_df, "semantic")
    
    norm_reference_df = aggregate_reference_metrics(metric_df)
    norm_reference_df = norm_reference_df[["pipeline", "metric", "normalized"]]
    # print(norm_reference_df.to_string())
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