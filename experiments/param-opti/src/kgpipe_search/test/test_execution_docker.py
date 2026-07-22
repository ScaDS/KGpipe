from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import uuid
from pathlib import Path

import pytest
from docker import DockerClient

from kgpipe_search.mounts import DEFAULT_SCRATCH_HOST, ScratchMount


def _swarm_available() -> tuple[bool, str]:
    try:
        client = DockerClient.from_env()
        info = client.info()
    except Exception as e:
        return False, f"Docker engine not reachable: {e}"

    swarm = info.get("Swarm") or {}
    state = (swarm.get("LocalNodeState") or "").lower()
    if state != "active":
        return False, f"Docker Swarm not active (LocalNodeState={swarm.get('LocalNodeState')!r})"

    return True, "ok"


_TEST_COMMAND = [
    "python",
    "-c",
    "import os, json, time; "
    "time.sleep(30); "
    "p=float(os.environ.get('KGPIPE_PARAM','0')); "
    "print(json.dumps({'result': p + 1.0}))",
]


def test_execute_pipeline_docker_swarm_parses_result():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    from kgpipe_search.execution import execute_pipeline_docker_swarm

    # The container prints {"result": float(KGPIPE_PARAM) + 1.0}
    param = "1.5"
    expected = 2.5

    result = execute_pipeline_docker_swarm(
        kg="dummy",
        source="dummy",
        config={
            "image": "python:3.12-slim",
            "command": _TEST_COMMAND,
            "parameter": param,
            "max_per_node": 1,
            "timeout_s": 120,
        },
    )

    assert result == expected


def test_extract_job_result_formats():
    from kgpipe_search.swarm import ResultSpec, SwarmJobResult, extract_job_result

    job = SwarmJobResult(
        service_id="svc",
        service_name="svc-name",
        run_name="run-1",
        node_id="node-1",
        state="complete",
        exit_code=0,
        logs='info line\n{"result": 2.5, "label": "ok"}\n',
    )

    assert extract_job_result(job, ResultSpec(format="float")) == 2.5
    assert extract_job_result(job, ResultSpec(format="json", json_key="label")) == "ok"
    assert extract_job_result(job, ResultSpec(format="json", json_key=None)) == {
        "result": 2.5,
        "label": "ok",
    }
    assert extract_job_result(job, ResultSpec(format="logs")) == job.logs
    assert extract_job_result(job, ResultSpec(format="exit_code")) == 0

    failed = SwarmJobResult(
        service_id="svc",
        service_name="svc-name",
        run_name="run-1",
        node_id="node-1",
        state="complete",
        exit_code=7,
        logs="",
    )
    assert extract_job_result(failed, ResultSpec(format="exit_code", require_exit_code=None)) == 7


def test_execute_pipeline_docker_swarm_exit_code_result():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    from kgpipe_search.execution import execute_pipeline_docker_swarm

    result = execute_pipeline_docker_swarm(
        kg="dummy",
        source="dummy",
        config={
            "image": "python:3.12-slim",
            "command": ["python", "-c", "import sys; sys.exit(42)"],
            "result_format": "exit_code",
            "require_exit_code": 42,
            "max_per_node": 1,
            "timeout_s": 120,
        },
    )

    assert result == 42


def test_execute_pipeline_docker_swarm_json_result():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    from kgpipe_search.execution import execute_pipeline_docker_swarm

    result = execute_pipeline_docker_swarm(
        kg="dummy",
        source="dummy",
        config={
            "image": "python:3.12-slim",
            "command": [
                "python",
                "-c",
                "import json; print(json.dumps({'metrics': {'f1': 0.9}, 'result': 0.9}))",
            ],
            "result_format": "json",
            "result_key": "metrics",
            "max_per_node": 1,
            "timeout_s": 120,
        },
    )

    assert result == {"f1": 0.9}


_SCRATCH_WRITE_COMMAND = [
    "python",
    "-c",
    "import json, os, pathlib; "
    "root = pathlib.Path(os.environ['KGPIPE_SCRATCH']) / os.environ['KGPIPE_RUN_ID']; "
    "root.mkdir(parents=True, exist_ok=True); "
    "(root / 'done.txt').write_text('ok'); "
    "print(json.dumps({'result': str(root / 'done.txt')}))",
]

_SCRATCH_VERIFY_COMMAND = [
    "python",
    "-c",
    "import os, pathlib, sys; "
    "p = pathlib.Path(os.environ['KGPIPE_SCRATCH']) / os.environ['KGPIPE_RUN_ID'] / 'done.txt'; "
    "sys.exit(0 if p.is_file() and p.read_text() == 'ok' else 1)",
]


def test_swarm_scratch_mount_writes_per_run_file():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    scratch_host = Path(os.environ.get("KGPIPE_SCRATCH_HOST", DEFAULT_SCRATCH_HOST))
    if not scratch_host.is_dir():
        pytest.skip(f"scratch host path does not exist: {scratch_host}")

    from kgpipe_search.swarm import ResultSpec, SwarmManager, extract_job_result

    mgr = SwarmManager()
    run_name = f"pytest-{uuid.uuid4().hex[:8]}"
    scratch = ScratchMount(host_path=str(scratch_host))

    res = mgr.run_job(
        image="python:3.12-slim",
        command=_SCRATCH_WRITE_COMMAND,
        run_name=run_name,
        scratch=scratch,
        max_per_node=1,
        timeout_s=120,
        name_prefix="kgpipe-scratch",
    )

    assert res.state == "complete"
    assert res.exit_code == 0
    assert res.run_name == run_name
    assert extract_job_result(res, ResultSpec(format="json", json_key="result")).endswith("done.txt")

    out_path = scratch_host / run_name / "done.txt"
    if out_path.is_file():
        assert out_path.read_text() == "ok"
        return

    assert res.node_id is not None
    verify = mgr.run_job(
        image="python:3.12-slim",
        command=_SCRATCH_VERIFY_COMMAND,
        run_name=run_name,
        scratch=scratch,
        node_id=res.node_id,
        max_per_node=1,
        timeout_s=120,
        name_prefix="kgpipe-scratch-verify",
    )
    assert verify.state == "complete"
    assert verify.exit_code == 0


def test_hdfs_copy_strategy_builds_put_command_with_hadoop_conf():
    from kgpipe_search.copy import CopyContext, CopyTarget, HdfsCopyStrategy

    strategy = HdfsCopyStrategy(hadoop_conf_host="/etc/hadoop/conf")
    plan = strategy.build_job(
        context=CopyContext(
            run_name="run-42",
            node_id="node-1",
            scratch=ScratchMount(
                host_path="/local/d1/docker-scratch",
                container_path="/local/d1/docker-scratch",
            ),
        ),
        destination=CopyTarget(path="/user/kgpipe/results"),
    )

    assert plan.image == "apache/hadoop:3.3.6"
    assert "hdfs dfs -put" in plan.command[-1]
    assert "/user/kgpipe/results/run-42" in plan.command[-1]
    assert plan.env["KGPIPE_HDFS_DEST"] == "/user/kgpipe/results/run-42"
    assert plan.env["HADOOP_CONF_DIR"] == "/etc/hadoop/conf"
    assert any(m.get("Source") == "/local/d1/docker-scratch" for m in plan.mounts)
    assert any(m.get("Source") == "/etc/hadoop/conf" for m in plan.mounts)


def test_hdfs_copy_strategy_builds_put_command_with_namenode_and_user():
    from kgpipe_search.copy import CopyContext, CopyTarget, HdfsCopyStrategy

    strategy = HdfsCopyStrategy(namenode="nn.example:8020", user="alice")
    plan = strategy.build_job(
        context=CopyContext(
            run_name="run-42",
            node_id="node-1",
            scratch=ScratchMount(
                host_path="/local/d1/docker-scratch",
                container_path="/local/d1/docker-scratch",
            ),
        ),
        destination=CopyTarget(path="/user/kgpipe/results"),
    )

    script = plan.command[-1]
    assert "fs.defaultFS" in script
    assert "hdfs://nn.example:8020" in script
    assert "hdfs dfs -put" in script
    assert plan.env["HADOOP_USER_NAME"] == "alice"
    assert plan.env["KGPIPE_HDFS_DEST"] == "hdfs://nn.example:8020/user/kgpipe/results/run-42"
    assert "HADOOP_CONF_DIR" not in plan.env
    assert any(m.get("Source") == "/etc/hosts" for m in plan.mounts)
    assert not any(m.get("Source") == "/etc/hadoop/conf" for m in plan.mounts)


def test_bind_copy_strategy_builds_cp_command_and_mounts():
    from kgpipe_search.copy import BindCopyStrategy, CopyContext, CopyTarget

    strategy = BindCopyStrategy()
    plan = strategy.build_job(
        context=CopyContext(
            run_name="run-42",
            node_id="node-1",
            scratch=ScratchMount(
                host_path="/local/d1/docker-scratch",
                container_path="/local/d1/docker-scratch",
            ),
        ),
        destination=CopyTarget(path="/u/hadena/shared-data/docker-results"),
    )

    assert plan.image == "alpine:3.20"
    assert "cp -a" in plan.command[-1]
    assert "run-42" in plan.command[-1]
    assert plan.env["KGPIPE_BIND_DEST"] == "/u/hadena/shared-data/docker-results"
    assert any(m.get("Source") == "/local/d1/docker-scratch" for m in plan.mounts)
    assert any(m.get("Source") == "/u/hadena/shared-data/docker-results" for m in plan.mounts)

# KGPIPE_HDFS_NAMENODE=athena1.informatik.intern.uni-leipzig.de KGPIPE_HDFS_USER=hadena KGPIPE_HDFS_DEST=/user/kgpipe/results \
#  uv run pytest -k swarm_hdfs_copy_stage -v

def test_swarm_bind_copy_stage():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    bind_dest = os.environ.get("KGPIPE_BIND_DEST_HOST")
    if not bind_dest:
        pytest.skip("set KGPIPE_BIND_DEST_HOST to run bind copy integration test")

    scratch_host = Path(os.environ.get("KGPIPE_SCRATCH_HOST", DEFAULT_SCRATCH_HOST))
    if not scratch_host.is_dir():
        pytest.skip(f"scratch host path does not exist: {scratch_host}")

    bind_dest_path = Path(bind_dest)
    if not bind_dest_path.is_dir():
        pytest.skip(f"bind destination host path does not exist: {bind_dest_path}")

    from kgpipe_search.copy import BindCopyStrategy, CopyTarget
    from kgpipe_search.mounts import ScratchMount
    from kgpipe_search.swarm import SwarmManager

    mgr = SwarmManager()
    run_name = f"pytest-bind-copy-{uuid.uuid4().hex[:8]}"
    scratch = ScratchMount(host_path=str(scratch_host))
    strategy = BindCopyStrategy()
    run = mgr.run_job_with_copy(
        image="python:3.12-slim",
        command=_SCRATCH_WRITE_COMMAND,
        run_name=run_name,
        scratch=scratch,
        copy_strategy=strategy,
        copy_destination=CopyTarget(path=str(bind_dest_path)),
        max_per_node=1,
        timeout_s=300,
        name_prefix="kgpipe-bind-copy",
    )
    assert run.job.state == "complete"
    assert run.job.exit_code == 0
    assert run.copy is not None, (
        f"copy stage missing; job logs:\n{run.job.logs}"
    )
    assert run.copy.state == "complete", (
        f"copy failed (exit_code={run.copy.exit_code}); copy logs:\n{run.copy.logs}"
    )
    assert run.copy.exit_code == 0
    assert run.copy.node_id == run.job.node_id
    assert "copied to" in run.copy.logs

    # If the bind dest is shared and visible on this host, check directly.
    out_path = bind_dest_path / run_name / "done.txt"
    if out_path.is_file():
        assert out_path.read_text() == "ok"
        return

    # Otherwise verify on the same node via a follow-up service.
    assert run.copy.node_id is not None
    verify = mgr.run_job(
        image="alpine:3.20",
        command=[
            "sh",
            "-lc",
            f"test -f /dst/{run_name}/done.txt && grep -qx ok /dst/{run_name}/done.txt",
        ],
        node_id=run.copy.node_id,
        mounts=[
            {"Type": "bind", "Source": str(bind_dest_path), "Target": "/dst"},
        ],
        max_per_node=1,
        timeout_s=120,
        name_prefix="kgpipe-bind-copy-verify",
    )
    assert verify.state == "complete"
    assert verify.exit_code == 0


def test_swarm_hdfs_copy_stage():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    hdfs_dest = os.environ.get("KGPIPE_HDFS_DEST")
    hdfs_namenode = os.environ.get("KGPIPE_HDFS_NAMENODE")
    hdfs_user = os.environ.get("KGPIPE_HDFS_USER")
    if not hdfs_dest:
        pytest.skip("set KGPIPE_HDFS_DEST to run HDFS copy integration test")

    hadoop_conf = os.environ.get("KGPIPE_HADOOP_CONF", "/etc/hadoop/conf")
    if hdfs_namenode is None and not Path(hadoop_conf).is_dir():
        pytest.skip(f"Hadoop config not found: {hadoop_conf}")

    scratch_host = Path(os.environ.get("KGPIPE_SCRATCH_HOST", DEFAULT_SCRATCH_HOST))
    if not scratch_host.is_dir():
        pytest.skip(f"scratch host path does not exist: {scratch_host}")

    from kgpipe_search.copy import CopyTarget, HdfsCopyStrategy
    from kgpipe_search.mounts import ScratchMount
    from kgpipe_search.swarm import SwarmManager

    if hdfs_namenode:
        copy_strategy = HdfsCopyStrategy(namenode=hdfs_namenode, user=hdfs_user)
    else:
        copy_strategy = HdfsCopyStrategy(hadoop_conf_host=hadoop_conf, user=hdfs_user)

    mgr = SwarmManager()
    run_name = f"pytest-hdfs-{uuid.uuid4().hex[:8]}"
    scratch = ScratchMount(host_path=str(scratch_host))

    run = mgr.run_job_with_copy(
        image="python:3.12-slim",
        command=_SCRATCH_WRITE_COMMAND,
        run_name=run_name,
        scratch=scratch,
        copy_strategy=copy_strategy,
        copy_destination=CopyTarget(path=hdfs_dest),
        max_per_node=1,
        timeout_s=300,
        name_prefix="kgpipe-hdfs-write",
    )

    assert run.job.state == "complete"
    assert run.job.exit_code == 0
    assert run.copy is not None, (
        f"copy stage missing; job logs:\n{run.job.logs}"
    )
    assert run.copy.state == "complete", (
        f"copy failed (exit_code={run.copy.exit_code}); copy logs:\n{run.copy.logs}"
    )
    assert run.copy.exit_code == 0
    assert run.copy.node_id == run.job.node_id
    assert "copied to" in run.copy.logs


#watch -n 1 'for s in $(docker service ls --format "{{.Name}}" | grep "^kgpipe-exp-"); do docker service ps "$s" --format "table {{.Name}}\t{{.Node}}\t{{.CurrentState}}\t{{.Error}}"; done'

def test_execute_pipeline_docker_swarm_parallel_on_all_nodes():
    ok, reason = _swarm_available()
    if not ok:
        pytest.skip(reason)

    from kgpipe_search.swarm import SwarmJobResult, SwarmManager, parse_job_result

    mgr = SwarmManager()
    node_ids = mgr.active_node_ids()
    if not node_ids:
        pytest.skip("No active Swarm nodes available")

    params = [str(float(i) + 1.0) for i in range(len(node_ids))]

    def run_job(param: str, target_node_id: str):
        res = mgr.run_job(
            image="python:3.12-slim",
            command=_TEST_COMMAND,
            parameter=param,
            node_id=target_node_id,
            max_per_node=1,
            timeout_s=120,
        )
        parsed = parse_job_result(res.logs)
        return param, target_node_id, res, parsed

    results: list[tuple[str, str, SwarmJobResult, float]] = []
    with ThreadPoolExecutor(max_workers=len(node_ids)) as pool:
        futures = [
            pool.submit(run_job, param, node_id)
            for param, node_id in zip(params, node_ids, strict=True)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    assert len(results) == len(node_ids)

    assigned_nodes: set[str] = set()
    for param, target_node_id, res, parsed in results:
        assert res.state == "complete"
        assert res.exit_code == 0
        assert parsed == float(param) + 1.0
        assert res.node_id == target_node_id
        assigned_nodes.add(target_node_id)

    assert assigned_nodes == set(node_ids), (
        f"expected one job per node ({len(node_ids)} nodes), "
        f"but jobs ran on {len(assigned_nodes)} distinct nodes"
    )