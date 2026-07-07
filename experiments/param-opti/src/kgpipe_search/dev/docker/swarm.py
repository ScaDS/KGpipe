from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from docker import DockerClient
from docker.errors import APIError, NotFound
from docker.models.services import Service
from docker.types import RestartPolicy, ServiceMode

from kgpipe_search.copy import CopyContext, CopyStrategy, CopyTarget
from kgpipe_search.mounts import (
    ScratchMount,
)
@dataclass(frozen=True)
class SwarmJobResult:
    service_id: str
    service_name: str
    run_name: str
    node_id: Optional[str]
    state: Literal["complete", "failed", "shutdown", "rejected", "orphaned"]
    exit_code: Optional[int]
    logs: str


@dataclass(frozen=True)
class SwarmRunResult:
    job: SwarmJobResult
    copy: SwarmJobResult | None = None


ResultFormat = Literal["float", "json", "logs", "exit_code"]


@dataclass(frozen=True)
class ResultSpec:
    """Describes what to return from a finished container job."""

    format: ResultFormat = "float"
    json_key: str | None = "result"
    require_exit_code: int | None = 0


def _json_lines(logs: str) -> list[Any]:
    parsed: list[Any] = []
    for line in logs.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed


def _value_from_json(obj: Any, key: str | None) -> Any:
    if key is None:
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object to read key {key!r}, got {type(obj).__name__}")
    if key not in obj:
        raise ValueError(f"JSON object has no key {key!r}")
    return obj[key]


def extract_job_result(job: SwarmJobResult, spec: ResultSpec | None = None) -> Any:
    """Extract the requested value from a finished Swarm job."""
    spec = spec or ResultSpec()

    if spec.require_exit_code is not None and job.exit_code != spec.require_exit_code:
        raise RuntimeError(
            f"expected exit code {spec.require_exit_code}, got {job.exit_code} "
            f"(state={job.state})"
        )

    if spec.format == "exit_code":
        if job.exit_code is None:
            raise ValueError("exit code unavailable")
        return job.exit_code

    if spec.format == "logs":
        return job.logs

    if spec.format == "json":
        lines = _json_lines(job.logs)
        if not lines:
            raise ValueError("no JSON found in logs")
        return _value_from_json(lines[-1], spec.json_key)

    if spec.format == "float":
        return float(_parse_scalar_from_logs(job.logs, key=spec.json_key))

    raise ValueError(f"unsupported result format: {spec.format!r}")


def _parse_scalar_from_logs(logs: str, *, key: str | None = "result") -> float | int | str:
    stripped = logs.strip()
    if not stripped:
        raise ValueError("empty logs")

    for obj in reversed(_json_lines(stripped)):
        value = _value_from_json(obj, key)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as e:
                raise ValueError(f"JSON key {key!r} is not numeric: {value!r}") from e
        raise ValueError(f"JSON key {key!r} is not numeric: {value!r}")

    return float(stripped.split()[-1])


def parse_job_result(logs: str) -> float:
    """Parse a float result from container stdout logs."""
    return float(_parse_scalar_from_logs(logs))


class SwarmManager:
    """
    Minimal Swarm "job runner" that launches one-shot services.

    Key feature: enforce a *per-node* cap by pinning each job to a node that has
    fewer than `max_per_node` active tasks with our label.
    """

    DEFAULT_APP_LABEL_KEY = "kgpipe.job"

    def __init__(self, *, app_label_key: str = DEFAULT_APP_LABEL_KEY, app_label_value: str = "1"):
        self.client = DockerClient.from_env()
        self._label_key = app_label_key
        self._label_value = app_label_value
        self._schedule_lock = threading.Lock()

    def active_node_ids(self) -> list[str]:
        node_ids: list[str] = []
        for node in self._nodes():
            node_id = node.get("ID")
            if not node_id:
                continue

            status_state = (((node.get("Status") or {}).get("State")) or "").lower()
            availability = (((node.get("Spec") or {}).get("Availability")) or "").lower()
            if status_state != "ready":
                continue
            if availability and availability != "active":
                continue

            node_ids.append(node_id)
        return node_ids

    def _nodes(self) -> Sequence[dict]:
        return self.client.api.nodes()

    def _active_task_counts_by_node(self) -> Dict[str, int]:
        """
        Count active tasks for our app label, grouped by NodeID.

        "Active" includes tasks that are accepted/starting/running/preparing,
        i.e. still occupying a container slot.
        """
        tasks = self.client.api.tasks(
            filters={
                "label": [f"{self._label_key}={self._label_value}"],
                "desired-state": ["running"],
            }
        )
        counts: Dict[str, int] = {}
        for t in tasks:
            node_id = t.get("NodeID")
            if not node_id:
                continue
            st = (((t.get("Status") or {}).get("State")) or "").lower()
            if st in {"new", "pending", "assigned", "accepted", "preparing", "starting", "running"}:
                counts[node_id] = counts.get(node_id, 0) + 1
        return counts

    def _pick_node_with_capacity(self, *, max_per_node: int) -> Optional[str]:
        nodes = self._nodes()
        if not nodes:
            return None

        counts = self._active_task_counts_by_node()

        eligible: list[Tuple[str, int]] = []
        for n in nodes:
            node_id = n.get("ID")
            if not node_id:
                continue

            status_state = (((n.get("Status") or {}).get("State")) or "").lower()
            availability = (((n.get("Spec") or {}).get("Availability")) or "").lower()
            if status_state != "ready":
                continue
            if availability and availability != "active":
                continue

            eligible.append((node_id, counts.get(node_id, 0)))

        if not eligible:
            return None

        eligible.sort(key=lambda x: x[1])
        node_id, used = eligible[0]
        if used >= max_per_node:
            return None
        return node_id

    def run_job(
        self,
        *,
        image: str,
        command: Optional[Sequence[str]] = None,
        args: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
        parameter: Optional[str] = None,
        node_id: Optional[str] = None,
        max_per_node: int = 1,
        timeout_s: int = 60 * 60,
        poll_interval_s: float = 1.0,
        cleanup: bool = True,
        extra_labels: Optional[Dict[str, str]] = None,
        name_prefix: str = "kgpipe-exp",
        run_name: Optional[str] = None,
        scratch: ScratchMount | None = None,
        mounts: Optional[list[dict]] = None,
    ) -> SwarmJobResult:
        """
        Launch a one-shot service (1 replica), wait for completion, fetch logs.

        The `parameter` is passed via env var `KGPIPE_PARAM` by default.
        When `scratch` is set, the host scratch directory is bind-mounted and
        `KGPIPE_RUN_ID` is set to `run_name` (default: job id) for per-run subdirs.
        """
        if max_per_node <= 0:
            raise ValueError("max_per_node must be >= 1")

        job_id = uuid.uuid4().hex[:12]
        run_id = run_name or job_id
        service_name = f"{name_prefix}-{job_id}"

        labels = {
            self._label_key: self._label_value,
            "kgpipe.job_id": job_id,
        }
        if extra_labels:
            labels.update(extra_labels)

        env_list: list[str] = []
        if env:
            env_list.extend([f"{k}={v}" for k, v in env.items()])
        if parameter is not None:
            env_list.append(f"KGPIPE_PARAM={parameter}")
        if scratch is not None:
            env_list.append(f"KGPIPE_RUN_ID={run_id}")
            env_list.append(f"KGPIPE_SCRATCH={scratch.container_path}")

        service_mounts = list(mounts or [])
        if scratch is not None:
            service_mounts.append(scratch.to_mount())

        deadline = time.time() + timeout_s
        last_err: Optional[Exception] = None

        service: Optional[Service] = None
        pinned_node_id: Optional[str] = None

        while time.time() < deadline and service is None:
            with self._schedule_lock:
                if node_id is not None:
                    counts = self._active_task_counts_by_node()
                    if counts.get(node_id, 0) >= max_per_node:
                        pinned_node_id = None
                    else:
                        pinned_node_id = node_id
                else:
                    pinned_node_id = self._pick_node_with_capacity(max_per_node=max_per_node)

                if pinned_node_id is None:
                    pass
                else:
                    mode = ServiceMode("replicated", replicas=1)
                    try:
                        service = self.client.services.create(
                            image=image,
                            command=list(command) if command else None,
                            args=list(args) if args else None,
                            env=env_list or None,
                            mounts=service_mounts or None,
                            name=service_name,
                            mode=mode,
                            labels=labels,
                            restart_policy=RestartPolicy(condition="none"),
                            constraints=[f"node.id=={pinned_node_id}"],
                            container_labels=labels,
                        )
                    except APIError as e:
                        last_err = e
                        service = None
                        pinned_node_id = None

            if service is None:
                time.sleep(poll_interval_s)

        if service is None:
            raise RuntimeError("Unable to schedule job before timeout") from last_err

        try:
            result = self._wait_service_done(
                service,
                timeout_s=max(1, int(deadline - time.time())),
                poll_interval_s=poll_interval_s,
            )
        finally:
            if cleanup:
                try:
                    service.remove()
                except Exception:
                    pass

        return SwarmJobResult(
            service_id=service.id,
            service_name=service_name,
            run_name=run_id,
            node_id=pinned_node_id,
            state=result["state"],
            exit_code=result.get("exit_code"),
            logs=result.get("logs", ""),
        )

    def copy_results(
        self,
        job: SwarmJobResult,
        *,
        scratch: ScratchMount,
        strategy: CopyStrategy,
        destination: CopyTarget,
        max_per_node: int = 1,
        timeout_s: int = 60 * 60,
        poll_interval_s: float = 1.0,
        cleanup: bool = True,
        name_prefix: str = "kgpipe-copy",
    ) -> SwarmJobResult:
        """
        Launch a copy service on the same node as `job`, moving scratch outputs
        to `destination` using the given copy strategy (e.g. HDFS).
        """
        if job.node_id is None:
            raise ValueError("cannot copy results: source job has no node_id")
        if job.state != "complete" or job.exit_code != 0:
            raise RuntimeError(
                f"cannot copy results from unsuccessful job "
                f"(state={job.state}, exit_code={job.exit_code})"
            )

        plan = strategy.build_job(
            context=CopyContext(
                run_name=job.run_name,
                node_id=job.node_id,
                scratch=scratch,
            ),
            destination=destination,
        )

        return self.run_job(
            image=plan.image,
            command=list(plan.command),
            env=plan.env,
            mounts=plan.mounts,
            node_id=job.node_id,
            run_name=job.run_name,
            max_per_node=max_per_node,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            cleanup=cleanup,
            name_prefix=name_prefix,
        )

    def run_job_with_copy(
        self,
        *,
        copy_strategy: CopyStrategy,
        copy_destination: CopyTarget,
        scratch: ScratchMount,
        copy_on_success: bool = True,
        **job_kwargs: Any,
    ) -> SwarmRunResult:
        """Run an experiment job and optionally copy its scratch outputs afterward."""
        job = self.run_job(scratch=scratch, **job_kwargs)
        if not copy_on_success:
            return SwarmRunResult(job=job)

        if job.state != "complete" or job.exit_code != 0:
            return SwarmRunResult(job=job)

        copy_job = self.copy_results(
            job,
            scratch=scratch,
            strategy=copy_strategy,
            destination=copy_destination,
            max_per_node=job_kwargs.get("max_per_node", 1),
            timeout_s=job_kwargs.get("timeout_s", 60 * 60),
            poll_interval_s=job_kwargs.get("poll_interval_s", 1.0),
            cleanup=job_kwargs.get("cleanup", True),
        )
        return SwarmRunResult(job=job, copy=copy_job)

    def _wait_service_done(
        self, service: Service, *, timeout_s: int, poll_interval_s: float
    ) -> Dict[str, Any]:
        deadline = time.time() + timeout_s

        def collect_logs() -> str:
            try:
                raw = service.logs(stdout=True, stderr=True)
                if raw is None:
                    return ""
                if isinstance(raw, (bytes, bytearray)):
                    return bytes(raw).decode("utf-8", errors="replace")
                chunks: list[bytes] = []
                for c in raw:
                    if isinstance(c, (bytes, bytearray)):
                        chunks.append(bytes(c))
                    else:
                        chunks.append(str(c).encode("utf-8", errors="replace"))
                return b"".join(chunks).decode("utf-8", errors="replace")
            except Exception:
                return ""

        while time.time() < deadline:
            try:
                tasks = service.tasks()
            except (APIError, NotFound):
                return {"state": "orphaned", "exit_code": None, "logs": ""}

            if not tasks:
                time.sleep(poll_interval_s)
                continue

            t = tasks[0]
            status = (t.get("Status") or {})
            state = (status.get("State") or "").lower()

            if state in {"complete", "failed", "shutdown", "rejected"}:
                exit_code = None
                container_status = status.get("ContainerStatus") or {}
                if "ExitCode" in container_status:
                    exit_code = container_status.get("ExitCode")

                logs = collect_logs()

                return {"state": state, "exit_code": exit_code, "logs": logs}

            time.sleep(poll_interval_s)

        logs = collect_logs()
        return {"state": "shutdown", "exit_code": None, "logs": logs}
