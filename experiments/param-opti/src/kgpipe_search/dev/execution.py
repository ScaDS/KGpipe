# Wrapper for pipeline execution and evaluation

from __future__ import annotations

from typing import Any, Dict

from kgpipe_search.copy import (
    copy_strategy_from_config,
    copy_target_from_config,
)
from kgpipe_search.mounts import (
    DEFAULT_SCRATCH_CONTAINER,
    DEFAULT_SCRATCH_HOST,
    ScratchMount,
)
from kgpipe_search.swarm import ResultSpec, SwarmManager, extract_job_result

type KG = str
type Source = str

type ConfigSpace = Dict[str, Any]
type Config = Dict[str, Any]

type EvaluationResult = float

_swarm = SwarmManager()


def _result_spec_from_config(config: Config) -> ResultSpec:
    require_exit_code = config.get("require_exit_code", 0)
    if require_exit_code == "any":
        require_exit_code = None

    return ResultSpec(
        format=config.get("result_format", "float"),
        json_key=config.get("result_key", "result"),
        require_exit_code=require_exit_code,
    )


def _scratch_from_config(config: Config) -> ScratchMount | None:
    scratch = config.get("scratch")
    if scratch is None:
        return None
    if isinstance(scratch, ScratchMount):
        return scratch
    if isinstance(scratch, str):
        return ScratchMount(host_path=scratch)
    if isinstance(scratch, dict):
        return ScratchMount(
            host_path=scratch.get("host_path", DEFAULT_SCRATCH_HOST),
            container_path=scratch.get("container_path", DEFAULT_SCRATCH_CONTAINER),
        )
    raise ValueError(f"invalid scratch config: {scratch!r}")


def execute_pipeline(kg: KG, source: Source, config: Config) -> float:
    pass


def execute_pipeline_docker(kg: KG, source: Source, config: Config) -> float:
    pass


def execute_pipeline_docker_swarm(kg: KG, source: Source, config: Config) -> Any:
    """
    Execute a single experiment in Swarm and return a result.

    Expected `config` keys (minimal):
    - image: str (required)
    - parameter: str (optional) passed via `KGPIPE_PARAM`

    Result handling:
    - result_format: "float" | "json" | "logs" | "exit_code" (default "float")
    - result_key: JSON field to read for "float"/"json" (default "result")
    - require_exit_code: expected exit code (default 0); use "any" to skip check

    Scratch (optional):
    - scratch: host path str, or dict with host_path/container_path
    - run_name: per-run subdirectory under scratch (default: auto job id)

    Copy (optional, requires scratch):
    - copy: destination path str, or dict with strategy/destination
      HDFS strategy supports either namenode+user or hadoop_conf_host

    Float format contract:
    - Container prints JSON with a numeric `result` field, or a bare float
      as the last token in logs.
    """
    scratch = _scratch_from_config(config)
    copy_cfg = config.get("copy")
    job_kwargs = {
        "image": config["image"],
        "command": config.get("command"),
        "args": config.get("args"),
        "env": config.get("env"),
        "parameter": config.get("parameter"),
        "max_per_node": int(config.get("max_per_node", 1)),
        "timeout_s": int(config.get("timeout_s", 60 * 60)),
        "extra_labels": {"kgpipe.kg": str(kg), "kgpipe.source": str(source)},
        "run_name": config.get("run_name"),
        "scratch": scratch,
        "mounts": config.get("mounts"),
    }

    if copy_cfg is not None:
        if scratch is None:
            raise ValueError("copy requires scratch to be configured")
        if isinstance(copy_cfg, str):
            strategy = copy_strategy_from_config({"type": "hdfs"})
            destination = copy_target_from_config(copy_cfg)
        else:
            strategy = copy_strategy_from_config(copy_cfg.get("strategy", {"type": "hdfs"}))
            destination = copy_target_from_config(
                copy_cfg.get("destination", copy_cfg.get("path"))
            )
        run = _swarm.run_job_with_copy(
            copy_strategy=strategy,
            copy_destination=destination,
            scratch=scratch,
            **job_kwargs,
        )
        res = run.job
        if run.copy is not None and run.copy.exit_code not in (0, None):
            raise RuntimeError(
                f"Swarm copy stage failed (state={run.copy.state}, exit_code={run.copy.exit_code})"
            )
    else:
        res = _swarm.run_job(**job_kwargs)

    try:
        return extract_job_result(res, _result_spec_from_config(config))
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(
            f"Swarm job result extraction failed "
            f"(state={res.state}, exit_code={res.exit_code})"
        ) from e
