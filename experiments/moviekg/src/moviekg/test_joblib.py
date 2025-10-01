from joblib import Memory, Parallel, delayed, hash as jhash
from pathlib import Path
import os

memory = Memory(location="cache/joblib", verbose=0)

def cache_key(pipeline_cfg, metric_cfg, data_snapshot_id, code_versions):
    # include only things that influence the output!
    return jhash({
        "pipe": pipeline_cfg,          # dict or frozen dataclass
        "metric": metric_cfg,
        "data": data_snapshot_id,      # e.g., delta/lakeFS/DVC commit id
        "code": code_versions,         # {"etl": git_sha1, "metrics": git_sha1}
    })



@memory.cache(ignore=["_logs_dir"])   # keep logs out of the hash
def compute_score(pipeline_cfg, metric_cfg, data_snapshot_id, code_versions, _logs_dir=None):
    # 1) run (or load) ETL output for this pipeline_cfg + snapshot
    #    ideally reuse pre-materialized datasets to avoid re-extracting per metric
    # df = run_pipeline(pipeline_cfg, data_snapshot_id)  # must be deterministic
    # # 2) compute metric
    # score = metric_fn(metric_cfg, df)
    print(f"Computing score for {pipeline_cfg}, {metric_cfg}, {data_snapshot_id}, {code_versions}")
    score = 1.0
    return {"score": float(score)}

def plan_tasks(pipelines, metrics, data_snapshot_id, code_versions):
    for p in pipelines:
        for m in metrics:
            yield (p, m, data_snapshot_id, code_versions)

def run_all(pipelines, metrics, data_snapshot_id, code_versions, n_jobs=-1):
    tasks = list(plan_tasks(pipelines, metrics, data_snapshot_id, code_versions))
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes", batch_size="auto")(
        delayed(compute_score)(p, m, data_snapshot_id, code_versions) for (p, m, _, _) in tasks
    )
    return results

def iterate_cache(mem):
    """Return the list of inputs and outputs from `mem` (joblib.Memory cache)."""
    for item in mem.store_backend.get_items():
        path_to_item = os.path.split(os.path.relpath(item.path, start=mem.store_backend.location))
        result = mem.store_backend.load_item(path_to_item)
        input_args = mem.store_backend.get_metadata(path_to_item).get("input_args")
        yield input_args, result

if __name__ == "__main__":
    # show cache
    for input_args, result in iterate_cache(memory):
        print(input_args, result)


    # pipelines = ["pipeline1", "pipeline2"]
    # metrics = ["metric1", "metric2"]
    # data_snapshot_id = "data_snapshot_id"
    # code_versions = {"etl": "1", "metrics": "metrics_version"}
    # results = run_all(pipelines, metrics, data_snapshot_id, code_versions)
    # print(results)