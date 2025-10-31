from matplotlib.font_manager import font_scalings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from typing import Dict
import re
import pandas as pd

import pandas as pd
from typing import List, Optional

from moviekg.paper.config import HEADERS, main_classes
from moviekg.pipelines.test_inc_ssp import pipeline_types, llm_pipeline_types


def load_metrics_from_file(file_path):
    # print("Loading metrics from file: ", file_path)
    df = pd.read_csv(file_path, names=HEADERS, skiprows=1)
    return df

def plot_growth_v1(df, metrics):
    """
    df: pandas DataFrame with columns:
        pipeline, stage, aspect, metric, value, normalized, details
    metrics: list[str] of metric names to plot

    Generates a subplot for each metric.
    Each subplot has x-axis: stage, y-axis: value.
    Each pipeline's value is a grouped bar at each stage.
    Returns (fig, axes).
    """
    required_cols = {"pipeline", "stage", "aspect", "metric", "value", "normalized", "details"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    if not isinstance(metrics, (list, tuple)) or len(metrics) == 0:
        raise ValueError("`metrics` must be a non-empty list of metric names.")

    # Only keep rows for requested metrics
    plot_df = df[df["metric"].isin(metrics)].copy()
    if plot_df.empty:
        raise ValueError("No rows found for the requested metrics.")

    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, max(3.5, 2.8 * n_metrics)), squeeze=False)
    axes = axes.ravel()

    # Overall (stable) pipeline order: alphabetical for consistency
    all_pipelines = sorted(plot_df["pipeline"].dropna().unique().tolist())

    for ax, metric in zip(axes, metrics):
        mdf = plot_df[plot_df["metric"] == metric].copy()
        if mdf.empty:
            ax.set_visible(False)
            continue

        # Preserve stage order as first-appearance order for this metric
        stage_order = pd.Index(mdf["stage"].dropna().astype(str)).drop_duplicates().tolist()
        if not stage_order:
            ax.set_visible(False)
            continue

        # Pivot to stage x pipeline = values
        pivot = (
            mdf.assign(stage=pd.Categorical(mdf["stage"].astype(str), categories=stage_order, ordered=True))
               .pivot_table(
                    index="stage",
                    columns="pipeline",
                    values="value",
                    aggfunc="sum",
               )
               .reindex(columns=all_pipelines)  # ensure consistent pipeline order
               .sort_index()
        )

        # If some pipelines/stages don't exist, fill with 0 (or use NaN if you prefer gaps)
        vals = pivot.fillna(0.0).values
        stages = pivot.index.astype(str).tolist()
        pipelines = pivot.columns.astype(str).tolist()

        n_stages = len(stages)
        n_pipes = max(1, len(pipelines))

        x = np.arange(n_stages, dtype=float)
        total_width = 0.8
        bar_w = total_width / n_pipes

        # Center the grouped bars around each stage tick
        start = x - (total_width / 2) + (bar_w / 2)

        for i, pipe in enumerate(pipelines):
            y = pivot[pipe].fillna(0.0).to_numpy()
            ax.bar(start + i * bar_w, y, width=bar_w, label=pipe)

        ax.set_title(str(metric))
        ax.set_xlabel("stage")
        ax.set_ylabel("value")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=0, ha="center")

        # Only show legend if multiple pipelines
        if n_pipes > 1:
            ax.legend(title="pipeline", frameon=False, ncols=min(3, n_pipes))
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    fig.tight_layout()
    return fig, axes

# --- Hardcoded pipeline colors (light/dark for solos; mid-tone for combined)
PALETTE = {
    # JSON solo
    "json_a": "#9ecae1", "json_b": "#1f77b4", "json_c": "21f77b4",
    # "json_baseA": "#9ecae1",
    # RDF solo
    "rdf_a":  "#a1d99b", "rdf_b":  "#2ca02c", "rdf_c": "#3ca02c",
    # TEXT solo
    "text_a": "#fdd0a2", "text_b": "#ff7f0e", "text_c": "#ff7f0e",

    # JSON mixed → violet
    "json_rdf_text": "#756bb1", "json_text_rdf": "#756bb1",
    # RDF mixed → teal
    "rdf_json_text":  "#1c9099", "rdf_text_json":  "#1c9099",
    # TEXT mixed → red-brown
    "text_json_rdf":  "#d95f0e", "text_rdf_json":  "#d95f0e",
}

HUE_ORDER = [
    "json_a","json_b","json_rdf_text","json_text_rdf",
    "rdf_a","rdf_b","rdf_json_text","rdf_text_json",
    "text_a","text_b","text_json_rdf","text_rdf_json"
]

def plot_growth(df, metrics, kind="bar", references={}):
    """
    df: pandas DataFrame with columns:
        pipeline, stage, aspect, metric, value, normalized, details
    metrics: list[str] of metric names to plot
    kind: "bar" or "line"
    
    Generates a facet plot (subplot per metric).
    Each subplot has x-axis: stage, y-axis: value,
    with different pipelines distinguished by color.
    """
    required_cols = {"pipeline", "stage", "aspect", "metric", "value", "normalized", "details"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    if not metrics:
        raise ValueError("`metrics` must be a non-empty list of metric names.")

    # Filter to requested metrics
    plot_df = df[df["metric"].isin(metrics)].copy()
    if plot_df.empty:
        raise ValueError("No rows found for the requested metrics.")

    # Consistent style
    sns.set(style="whitegrid")

    stage_order = list(dict.fromkeys(plot_df["stage"]))

    # sns.set_context("notebook", font_scale=1.2)

    # Facet grid WITHOUT hue to avoid legend kwarg collisions
    g = sns.FacetGrid(
        plot_df,
        col="metric",
        col_wrap=len(metrics),
        height=len(metrics)*1.6,
        aspect=1.5,
        sharey=False,
        col_order=metrics,
        legend_out=False,
    )

    if kind != "bar":
        raise ValueError("`kind` must be 'bar' for per-bar labels.")

    # Draw grouped bars with hue specified inside map_dataframe
    g.map_dataframe(
        sns.barplot,
        x="stage",
        y="value",
        hue="pipeline",
        hue_order=HUE_ORDER,
        palette=PALETTE,
        order=stage_order,
        dodge=True,
        errorbar=None
    )


    try:
        g._legend.remove()
    except Exception:
        pass

    # build a single combined legend below everything
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(6, len(labels)),   # 6 items per row (→ 2 rows for 12 pipelines)
        bbox_to_anchor=(0.5, -0.1), # adjust vertical offset
        frameon=False
    )

    for ax_idx, ax in enumerate(g.axes.flat):

        # remove x axis label
        ax.set_xlabel("")

        # numbers 1 to 3
        for stage_idx in range(1, 4):
            value, nvalue, details = get_reference_value(df, metrics[ax_idx], "stage_"+str(stage_idx))
            # print(metrics[ax_idx], value)
            xpos = stage_idx
            if stage_idx == 0:
                ax.axhline(value, ls="--", color="red")
            else:
                ax.axhline(value, ls="--", color="black")

    for ax in g.axes.flat:
        ax.set_xlabel("")
        # tidy up axes
        ax.set_xticks(range(len(stage_order)))
        ax.set_xticklabels(stage_order)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.margins(x=0.02)

    return g

def _stage_sort_key(s):
    """
    Convert 'stage_3' -> 3 for natural sorting; unknown formats go to +inf.
    """
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else float("inf")

def _shorten_iri(iri):
    """
    Turn 'http://kg.org/ontology/Person' -> 'Person' for cleaner legends.
    """
    return str(iri).rstrip("/").split("/")[-1]

def _flatten_to_df(nested):
    """
    nested: dict like {
      'rdf_a': {'stage_1': {'iri': count, ...}, ...},
      'reference': {...},
      ...
    }
    Returns a tidy DataFrame with columns:
      Pipeline, Stage, Class, Actual, Expected
    """

    # Split out reference (Expected) from others (Actual)
    if "reference" not in nested:
        raise ValueError("Input must contain a 'reference' key with expected counts.")
    ref = nested["reference"]
    pipelines = {k: v for k, v in nested.items() if k != "reference"}

    # Collect all stages/classes across data to ensure aligned zeros
    all_stages = sorted(
        {s for d in nested.values() for s in d.keys()},
        key=_stage_sort_key
    )



    all_classes = sorted(
        {c for d in nested.values() for s in d.values() for c in s.keys()}
    )



    # Build rows
    rows = []
    for pipe, pdata in pipelines.items():
        for stage in all_stages:
            for cls in all_classes:
                actual = pdata.get(stage, {}).get(cls, 0)
                expected = ref.get(stage, {}).get(cls, 0)
                if cls not in main_classes:
                    cls = "Other"
                rows.append({
                    "Pipeline": pipe,
                    "Stage": stage,
                    "Class": cls,
                    "Actual": actual,
                    "Expected": expected,
                    "Class Short": _shorten_iri(cls),
                })
    return pd.DataFrame(rows), [ _shorten_iri(c) for c in all_classes ], all_stages, list(pipelines.keys())

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def plot_actual_expected_stacked(df,
                                 pipeline_order=None,
                                 stage_order=None,
                                 class_order=None,
                                 col_wrap=3,
                                 height=4,
                                 suptitle="Actual vs Expected (stacked by Class) per Pipeline & Stage"):
    # --- prep ---
    df = df.copy()
    # ensure numeric & fill NAs
    for col in ["Actual", "Expected"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use Class Short as plotting label (cleaner legend)
    if "Class Short" not in df.columns:
        df["Class Short"] = df["Class"]

    # Default orders (preserve first-seen order)
    if pipeline_order is None:
        pipeline_order = list(pd.unique(df["Pipeline"]))
    if stage_order is None:
        stage_order = list(pd.unique(df["Stage"]))
    if class_order is None:
        class_order = list(pd.unique(df["Class Short"]))

    # aggregate once
    gdf = (
        df.groupby(["Pipeline", "Stage", "Class Short"], as_index=False)
          .agg(Actual=("Actual","sum"), Expected=("Expected","sum"))
    )

    # full grid to align missing combos to 0
    full_index = pd.MultiIndex.from_product(
        [pipeline_order, stage_order], names=["Pipeline","Stage"]
    )

    # pivots: (Pipeline, Stage) × Class
    actual = (gdf.pivot_table(index=["Pipeline","Stage"], columns="Class Short",
                              values="Actual", aggfunc="sum")
                .reindex(full_index)
                .reindex(columns=class_order)
                .fillna(0))
    expected = (gdf.pivot_table(index=["Pipeline","Stage"], columns="Class Short",
                                values="Expected", aggfunc="sum")
                  .reindex(full_index)
                  .reindex(columns=class_order)
                  .fillna(0))

    # --- plot ---
    sns.set(style="whitegrid")
    n_pipes = len(pipeline_order)
    ncols = min(col_wrap, n_pipes)
    nrows = (n_pipes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*height*1.6, nrows*height), squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    # palettes
    blues = sns.color_palette("Blues", n_colors=max(3, len(class_order)))
    oranges = sns.color_palette("Oranges", n_colors=max(3, len(class_order)))
    color_map_actual = {cls: blues[i % len(blues)] for i, cls in enumerate(class_order)}
    color_map_expected = {cls: oranges[i % len(oranges)] for i, cls in enumerate(class_order)}

    width = 0.4
    for ax, pipeline in zip(axes, pipeline_order):
        act = actual.loc[pipeline]   # index=Stage, cols=Class Short
        exp = expected.loc[pipeline] # index=Stage, cols=Class Short

        x = range(len(stage_order))

        # stacked bars
        bottom_a = [0.0]*len(stage_order)
        bottom_e = [0.0]*len(stage_order)

        for cls in class_order:
            a_vals = act[cls].to_numpy()
            e_vals = exp[cls].to_numpy()

            ax.bar([xi - 0.2 for xi in x], a_vals, width=width, bottom=bottom_a, color=color_map_actual[cls], edgecolor="none", label="Actual")
            ax.bar([xi + 0.2 for xi in x], e_vals, width=width, bottom=bottom_e, color=color_map_expected[cls], edgecolor="none", label="Expected")

            # update bottoms
            bottom_a = [b + v for b, v in zip(bottom_a, a_vals)]
            bottom_e = [b + v for b, v in zip(bottom_e, e_vals)]

        # cosmetics
        ax.set_title(pipeline)
        ax.set_xticks(list(x))
        ax.set_xticklabels(stage_order)
        ax.set_xlabel("Stage")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    # hide any unused axes
    for j in range(len(pipeline_order), len(axes)):
        fig.delaxes(axes[j])

    # legend
    handles = (
        [Patch(facecolor=color_map_actual[c], label=f"{c} • Actual") for c in class_order] +
        [Patch(facecolor=color_map_expected[c], label=f"{c} • Expected") for c in class_order]
    )

    # legend (robust placement)
    ncol_leg = min(4, len(handles))
    nrows_leg = int(np.ceil(len(handles) / ncol_leg))

    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol_leg,
        bbox_to_anchor=(0.5, 0.02),   # inside the figure, just above bottom
        frameon=False
    )

    # Title inside the top of the figure
    fig.suptitle(suptitle, y=0.99, fontsize=14)

    # Give the legend guaranteed space at the bottom, proportional to its rows
    # (works alongside constrained_layout)
    plt.subplots_adjust(bottom=0.08 + 0.05 * max(0, nrows_leg - 1))

    return fig


def plot_expected_actual_from_nested(
    nested,
    col_wrap=3,
    height=4,
    suptitle="Actual vs Expected (stacked by Class) per Pipeline & Stage"
):
    """
    nested: dict structured like the user's example.
    Creates one subplot per pipeline. For each Stage on that subplot,
    draws two stacked bars (Actual & Expected), each stacked by Class.
    """

    df, class_labels, stage_order, pipeline_order = _flatten_to_df(nested)

    # We’ll use the *short* class labels for stacking order & legend
    classes = class_labels

    # Prepare nice style
    sns.set(style="whitegrid")
    g = sns.FacetGrid(
        df,
        col="Pipeline",
        col_wrap=col_wrap,
        height=height,
        sharey=True,
        col_order=pipeline_order
    )

    df[['Actual','Expected']] = df[['Actual','Expected']].fillna(0)

    # Aggregate by Pipeline, Stage, Class, and Class Short
    df = (
        df.groupby(['Pipeline', 'Stage', 'Class', 'Class Short'], as_index=False)
        .agg({'Actual': 'sum', 'Expected': 'sum'})
    )

    return plot_actual_expected_stacked(df, pipeline_order, stage_order, ["Other", "Person", "Company", "Film"], col_wrap, height, suptitle)


def plot_class_occurence(df):
    """
    df: pandas dataframe with columns: pipeline, stage, aspect, metric, value, normalized, details
    """

    # filter df for metrics
    df = df[df["metric"].isin(["class_occurrence"])]
    # filter details contains unique_classes
    df = df[df["details"].str.contains("unique_classes")]
    # remove duration column
    df = df.drop(columns=["duration"])
    # filter not seed pipeline
    df = df[df["pipeline"] != "seed"]
    

    class_counts_by_stage_by_pipeline = {}
    # for each row
    for index, row in df.iterrows():
        details = json.loads(row["details"])
        classes = details["classes"]
        if row["pipeline"] not in class_counts_by_stage_by_pipeline:
            class_counts_by_stage_by_pipeline[row["pipeline"]] = {}
        # skip stage 0
        if row["stage"] == "stage_0":
            continue
        if row["stage"] not in class_counts_by_stage_by_pipeline[row["pipeline"]]:
            class_counts_by_stage_by_pipeline[row["pipeline"]][row["stage"]] = {}
        for class_name, count in classes.items():
            if class_name not in class_counts_by_stage_by_pipeline[row["pipeline"]][row["stage"]]:
                class_counts_by_stage_by_pipeline[row["pipeline"]][row["stage"]][class_name] = 0
            class_counts_by_stage_by_pipeline[row["pipeline"]][row["stage"]][class_name] += count

    # remove stage_0
    class_counts_by_stage_by_pipeline = {k: v for k, v in class_counts_by_stage_by_pipeline.items() if k != "stage_0"}

    return plot_expected_actual_from_nested(class_counts_by_stage_by_pipeline, col_wrap=2, height=4, suptitle="Actual vs Reference by Stage • Stacked by Class")


def rank_pipeline_stage(group_df, metric_names, metric_weights):
    weights = pd.Series(metric_weights, index=metric_names)
    vals = (
        group_df.set_index("metric")["normalized"]
        .reindex(metric_names)          # align order
        .astype(float)
    )
    return float((vals * weights).sum()/len(vals))

def rank_metrics_apply(df, metric_names, metric_weights):
    dff = df[df["metric"].isin(metric_names)]
    return (
        dff.groupby(["pipeline", "stage"])
           .apply(lambda g: rank_pipeline_stage(g, metric_names, metric_weights))
           .rename("score")
           .reset_index()
    )


def rank_metrics(
    df: pd.DataFrame,
    metric_names: List[str],
    metric_weights: List[float],
    *,
    agg: str = "mean",
    fill_missing: Optional[float] = 0.0,
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Compute a weighted score per (pipeline, stage) using normalized metric values.

    Parameters
    ----------
    df : DataFrame
        Must include columns: pipeline, stage, metric, normalized
        (other columns are ignored).
    metric_names : list of str
        Names of metrics to include, in the same order as their weights.
    metric_weights : list of float
        Weights aligned to metric_names.
    agg : {"mean","sum","max","min"}, default "mean"
        If there are duplicate rows per (pipeline, stage, metric), how to aggregate.
    fill_missing : float or None, default 0.0
        Value to fill when a metric is missing for a (pipeline, stage).
        Use None to leave as NaN (then the final score may be NaN).
    score_col : str, default "score"
        Name of the output score column.

    Returns
    -------
    DataFrame with columns: pipeline, stage, <score_col>
    """
    if len(metric_names) != len(metric_weights):
        raise ValueError("metric_names and metric_weights must have the same length")

    # Keep only what we need
    dff = df.loc[df["metric"].isin(metric_names), ["pipeline", "stage", "metric", "normalized"]]

    # Aggregate duplicates per (pipeline, stage, metric)
    agg_map = {"mean": "mean", "sum": "sum", "max": "max", "min": "min"}
    if agg not in agg_map:
        raise ValueError(f'agg must be one of {list(agg_map)}')
    pivot = dff.pivot_table(
        index=["pipeline", "stage"],
        columns="metric",
        values="normalized",
        aggfunc=agg_map[agg],
    )

    # Enforce column order and align with weights
    pivot = pivot.reindex(columns=metric_names)
    if fill_missing is not None:
        pivot = pivot.fillna(fill_missing)

    weights = pd.Series(metric_weights, index=metric_names)
    scores = pivot.dot(weights).rename(score_col)

    return scores.reset_index()

def get_reference_value(df, metric_name, stage):
    df = df[df["metric"] == metric_name]
    df = df[df["stage"] == stage]
    df = df[df["pipeline"] == "reference"]
    # print(df.to_string())
    value = df["value"].values[0]
    nvalue = df["normalized"].values[0]
    details = json.loads(df["details"].values[0])
    return value, nvalue, details


def get_reference_class_counts(df) -> Dict[str, Dict[str, int]]:
    df = df[df["pipeline"] == "reference"]
    reference_stage_class_count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    df = df[df["metric"] == "class_occurrence"]
    for stage in df["stage"].unique():
        df_stage = df[df["stage"] == stage]
        details = json.loads(df_stage["details"].values[0])
        class_counts = details["classes"]
        for class_name, count in class_counts.items():
            reference_stage_class_count[stage][class_name.split("/")[-1]] += count

    return reference_stage_class_count

# def subplot_source_entity_integration(df):
#     pass

from collections import defaultdict

def plot_class_occurence_new(df, reference_stage_class_count, classes):

    df = df[df["metric"] == "class_occurrence"]


    pipeline_stage_class_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    rows = []

    # for each pipeline and stage
    for pipeline in df["pipeline"].unique():
        for stage in df["stage"].unique():
            df_pipeline_stage = df[df["pipeline"] == pipeline]
            df_pipeline_stage = df_pipeline_stage[df_pipeline_stage["stage"] == stage]
            details = json.loads(df_pipeline_stage["details"].values[0])
            class_counts = details["classes"]
            for class_name, count in class_counts.items():
                if class_name not in classes:
                    class_name = "Other"
                pipeline_stage_class_count[pipeline][stage][class_name] += count

    # convert dict of dict to rows
    for pipeline, stage_class_count in pipeline_stage_class_count.items():
        for stage, class_count in stage_class_count.items():
            for class_name, count in class_count.items():
                rows.append({"pipeline": pipeline, "stage": stage, "class": class_name.split("/")[-1], "count": count})

    # df: pipeline, stage, class, count
    df = pd.DataFrame(rows)
    df = df[df["class"] != "Other"]

    classes_short = [class_name.split("/")[-1] for class_name in classes]

    sns.set(style="whitegrid")

    stage_order = list(dict.fromkeys(df["stage"]))
    g = sns.FacetGrid(
        df,
        col="class",
        col_wrap=3,
        height=4,
        aspect=1.5,
        sharey=False,
        col_order=classes_short #+["Other"],  # preserve requested order
    )
    g.map_dataframe(
        sns.barplot,
        x="stage",
        y="count",
        hue="pipeline",
        hue_order=HUE_ORDER,
        palette=PALETTE,
        order=stage_order,
        dodge=True,
        errorbar=None
    )


    for ax_idx, ax in enumerate(g.axes.flat):
        class_idx = ax_idx
        class_name = classes_short[class_idx]

        # remove x axis label
        ax.set_xlabel("")

        for stage, class_counts in reference_stage_class_count.items():
            xpos = int(stage.split("_")[1])
            if stage == "stage_0":
                ax.axhline(class_counts[class_name], ls="--", color="red")
            else:
                ax.axhline(class_counts[class_name], ls="--", color="black")


    # g.add_legend()

    if g.legend is not None:
        g.legend.remove()

    # build a combined legend below everything
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(6, len(labels)),   # split across columns
        bbox_to_anchor=(0.5, -0.02) # push below the grid
    )

    # make space at bottom so legend isn’t cut off
    g.fig.subplots_adjust(bottom=0.2)

    plt.subplots_adjust(top=0.88)

    # g.savefig("class_occurence_new.png")

    return g


def plot_class_occ_4_bar_chart(df):
    metrics = ["class_occurrence"]
    stages = ["stage_1", "stage_2", "stage_3"]
    all_reference_values = {}
    for metric in metrics:
        for stage in stages:
            value, nvalue, details = get_reference_value(df, metric, stage)
            all_reference_values[metric] = {
                "value": value,
                "nvalue": nvalue,
                "details": details
            }

    reference_stage_class_count = get_reference_class_counts(df)

    # remove seed and reference pipeline
    df = df[df["pipeline"] != "seed"]
    df = df[df["pipeline"] != "reference"]

    # subplot_source_entity_integration(df)

    classes = ["http://kg.org/ontology/Film", "http://kg.org/ontology/Person", "http://kg.org/ontology/Company"]


    return plot_class_occurence_new(df, reference_stage_class_count, classes)
