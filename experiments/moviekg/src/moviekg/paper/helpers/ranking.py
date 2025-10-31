import pandas as pd
from moviekg.config import OUTPUT_ROOT
from typing import Any, Mapping, List, Dict
from moviekg.paper.helpers.getter import (
    pipeline_stage_metric_dict, pipeline_name, metric_name, metric_value, 
    TABLE_DISPLAY_NAMES,
    normalize_metric, normalize_min_best, normalize_max_best,
    sta_fact_count, sta_denisity, sta_duration, #memory_peak is not considered
    ref_kg_p, ref_source_entity_f1, 
    sem_disjoint_domain, sem_incorrect_relation_direction, sem_incorrect_relation_range, sem_incorrect_relation_domain, sem_incorrect_datatype, sem_incorrect_datatype_format
)
from collections import defaultdict


type pipeline_agg = Mapping[pipeline_name, float]

def agg_metrics(psmd: pipeline_stage_metric_dict, metric_names: List[metric_name]) -> pipeline_agg:
    values_by_pipeline: Dict[pipeline_name, List[metric_value]] = defaultdict[pipeline_name, List[metric_value]](lambda: [])
    for pipeline, stage_dict in psmd.items():
        if pipeline in ["reference", "seed"]:
            continue
        for stage, metric_dict in stage_dict.items():
            if stage not in ["stage_3"]: # only stage 3 is considered
                continue
            for metric_name in metric_names:
                if metric_name in metric_dict:
                    values_by_pipeline[pipeline].append(metric_dict[metric_name])
                else:
                    print(f"pipeline: {pipeline}")
                    print(f"stage: {stage}")    
                    print(f"metric_names: {metric_names}")
                    print(f"metric_dict: {metric_dict}")
                    raise ValueError(f"Metric {metric_name} not found in metric_names")
        
    res: pipeline_agg = defaultdict[pipeline_name, float](lambda: 0.0)

    for pipeline, values in values_by_pipeline.items():
        filtered_values = [value for value in values if value >= 0]
        res[pipeline] = sum(filtered_values) / len(filtered_values)
        print(pipeline)
        print(" |\t".join(metric_names))
        print(" |\t".join([ str(value) for value in values_by_pipeline[pipeline]]))
        print("="+str(res[pipeline]))
        print("--------------------------------")

    return res

def _rank_and_save2csv(weights: dict, outfile_stem: str, psmd: pipeline_stage_metric_dict, round_digits: int = 3) -> None:
    
    # psmd = normalize_metric(psmd, sta_fact_count.__name__, ["stage_3"], normalize_max_best)
    psmd = normalize_metric(psmd, sta_denisity.__name__, ["stage_3"], normalize_max_best)
    sta_metric_names = [sta_denisity.__name__+"_norm"] # sta_fact_count.__name__+"_norm", 
    sta_agg = agg_metrics(psmd, sta_metric_names)

    sem_metric_names = [
        sem_disjoint_domain.__name__, sem_incorrect_relation_direction.__name__, 
        sem_incorrect_relation_range.__name__, sem_incorrect_relation_domain.__name__, 
        sem_incorrect_datatype.__name__, sem_incorrect_datatype_format.__name__]
    sem_agg = agg_metrics(psmd, sem_metric_names)
    
    ref_metric_names = [ref_kg_p.__name__, ref_source_entity_f1.__name__+"_avg", "ref_selected_task_metric_avg"]
    ref_agg = agg_metrics(psmd, ref_metric_names)

    psmd = normalize_metric(psmd, sta_duration.__name__+"_sum", ["stage_3"], normalize_min_best)
    eff_metric_names = [sta_duration.__name__+"_sum_norm"]
    eff_agg = agg_metrics(psmd, eff_metric_names)

    import json
    json.dump(psmd, open(OUTPUT_ROOT / f"paper/{outfile_stem}_psmd.json", "w"), indent=4)

    df_rows = []

    for pipeline, value in sem_agg.items():
        df_rows.append(
            {
                "pipeline": pipeline, 
                "semantic": round(value, round_digits), 
                "reference": round(ref_agg[pipeline], round_digits),
                "size": round(sta_agg[pipeline], round_digits),
                "efficiency": round(eff_agg[pipeline], round_digits)
            }
        )


    df = pd.DataFrame(df_rows)

    cols = ["size", "semantic", "reference", "efficiency"]
    # Ensure we only use known columns; fill missing weights with 0.0
    w = pd.Series(weights).reindex(cols, fill_value=0.0)

    # Compute combined score
    df = df[["pipeline"] + cols].copy()
    df["combined"] = (df[cols] * w).sum(axis=1).round(round_digits)

    print(df.to_string())

    # Sort & save (keep default index=True to match original behavior)
    out = df[["pipeline", "combined"]].sort_values(by="combined", ascending=False)
    out.to_csv(OUTPUT_ROOT / f"paper/{outfile_stem}.csv", sep="\t")

def _rank_and_save(weights: dict, outfile_stem: str, df: pd.DataFrame, round_digits: int = 3) -> None:
    """
    Compute weighted 'combined' score and save a TSV sorted by 'combined'.
    Uses the same behavior as your original functions (round to 3, keep default index in CSV).
    """
    cols = ["size", "semantic", "reference", "efficiency"]
    # Ensure we only use known columns; fill missing weights with 0.0
    w = pd.Series(weights).reindex(cols, fill_value=0.0)

    # Compute combined score
    df = df[["pipeline"] + cols].copy()
    df["combined"] = (df[cols] * w).sum(axis=1).round(round_digits)

    # Sort & save (keep default index=True to match original behavior)
    out = df[["pipeline", "combined"]].sort_values(by="combined", ascending=False)
    out.to_csv(OUTPUT_ROOT / f"paper/{outfile_stem}.csv", sep="\t")

