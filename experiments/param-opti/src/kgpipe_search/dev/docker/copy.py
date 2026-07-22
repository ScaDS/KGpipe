from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from kgpipe_search.mounts import ScratchMount

CopyStrategyName = Literal["hdfs", "bind"]

DEFAULT_HADOOP_CONF_HOST = "/etc/hadoop/conf"
DEFAULT_HADOOP_CONF_CONTAINER = "/etc/hadoop/conf"
MINIMAL_HADOOP_CONF_CONTAINER = "/tmp/kgpipe-hadoop-conf"
DEFAULT_HDFS_PORT = 9000
HOSTS_MOUNT_SOURCE = "/etc/hosts"
HOSTS_MOUNT_TARGET = "/etc/hosts"


def _normalize_namenode(namenode: str, *, default_port: int = DEFAULT_HDFS_PORT) -> str:
    raw = namenode.strip().rstrip("/")
    if raw.startswith("hdfs://"):
        authority, _, path = raw[len("hdfs://") :].partition("/")
        host = authority
    else:
        host, _, path = raw.partition("/")
        path = f"/{path}" if path else ""

    if ":" not in host:
        host = f"{host}:{default_port}"

    return f"hdfs://{host}{path}"


def _hosts_mount() -> dict[str, str]:
    return {
        "Type": "bind",
        "Source": HOSTS_MOUNT_SOURCE,
        "Target": HOSTS_MOUNT_TARGET,
        "ReadOnly": True,
    }


@dataclass(frozen=True)
class CopyTarget:
    """Destination path for a copy strategy (e.g. HDFS directory)."""

    path: str


@dataclass(frozen=True)
class CopyContext:
    """Source location on the node where the experiment wrote outputs."""

    run_name: str
    node_id: str
    scratch: ScratchMount


@dataclass(frozen=True)
class CopyJobPlan:
    image: str
    command: Sequence[str]
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[dict[str, str]] = field(default_factory=list)


class CopyStrategy(Protocol):
    def build_job(self, *, context: CopyContext, destination: CopyTarget) -> CopyJobPlan:
        ...


@dataclass(frozen=True)
class BindCopyStrategy:
    """
    Copy scratch outputs to a host bind-mounted directory using `cp`.

    `destination.path` must be a **host path on every node** (same path),
    e.g. an NFS mountpoint or any shared filesystem mounted consistently.
    """

    image: str = "alpine:3.20"
    dest_container_path: str = "/dst"
    scratch_container_path: str | None = None

    def build_job(self, *, context: CopyContext, destination: CopyTarget) -> CopyJobPlan:
        scratch_root = self.scratch_container_path or context.scratch.container_path
        src = f"{scratch_root}/{context.run_name}"
        dst = f"{self.dest_container_path}/{context.run_name}"

        command = [
            "sh",
            "-lc",
            (
                "set -euo pipefail; "
                f"test -d {src!r}; "
                f"mkdir -p {dst!r}; "
                f"cp -a {src!r}/. {dst!r}/; "
                f"echo copied to {dst!r}"
            ),
        ]

        mounts: list[dict[str, str]] = [
            context.scratch.to_mount(),
            {
                "Type": "bind",
                "Source": destination.path,
                "Target": self.dest_container_path,
            },
        ]

        env = {
            "KGPIPE_RUN_ID": context.run_name,
            "KGPIPE_SCRATCH": scratch_root,
            "KGPIPE_BIND_DEST": destination.path,
        }

        return CopyJobPlan(image=self.image, command=command, env=env, mounts=mounts)


def _hdfs_path(namenode: str | None, path: str, *, default_port: int = DEFAULT_HDFS_PORT) -> str:
    normalized = path.rstrip("/")
    if normalized.startswith("hdfs://"):
        return normalized
    if namenode is None:
        return normalized
    rel = normalized if normalized.startswith("/") else f"/{normalized}"
    return f"{_normalize_namenode(namenode, default_port=default_port)}{rel}"


def _minimal_hadoop_conf_script(*, namenode: str, conf_dir: str) -> str:
    nn = _normalize_namenode(namenode)
    return (
        f"mkdir -p {conf_dir!r}; "
        f"cat > {conf_dir!r}/core-site.xml <<'EOF'\n"
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<?xml-stylesheet type=\"text/xsl\" href=\"configuration.xsl\"?>\n"
        "<configuration>\n"
        "  <property>\n"
        "    <name>fs.defaultFS</name>\n"
        f"    <value>{nn}</value>\n"
        "  </property>\n"
        "</configuration>\n"
        "EOF\n"
        f"export HADOOP_CONF_DIR={conf_dir!r}"
    )


@dataclass(frozen=True)
class HdfsCopyStrategy:
    """
    Copy node-local scratch outputs to HDFS using the hdfs CLI.

    Configure either:
    - `namenode` (+ optional `user`) for a minimal client setup, or
    - `hadoop_conf_host` to mount cluster config from the node.
    """

    image: str = "apache/hadoop:3.3.6"
    namenode: str | None = None
    user: str | None = None
    hadoop_conf_host: str | None = None
    hadoop_conf_container: str = DEFAULT_HADOOP_CONF_CONTAINER
    minimal_conf_container: str = MINIMAL_HADOOP_CONF_CONTAINER
    mount_node_hosts: bool = True
    hdfs_port: int = DEFAULT_HDFS_PORT
    extra_env: dict[str, str] = field(default_factory=dict)

    def build_job(self, *, context: CopyContext, destination: CopyTarget) -> CopyJobPlan:
        src = f"{context.scratch.container_path}/{context.run_name}"
        hdfs_dst = _hdfs_path(self.namenode, destination.path, default_port=self.hdfs_port)
        hdfs_dst = f"{hdfs_dst}/{context.run_name}"

        setup_parts: list[str] = []
        env: dict[str, str] = {
            "KGPIPE_RUN_ID": context.run_name,
            "KGPIPE_SCRATCH": context.scratch.container_path,
            "KGPIPE_HDFS_DEST": hdfs_dst,
            **self.extra_env,
        }

        if self.user:
            env["HADOOP_USER_NAME"] = self.user

        if self.namenode is not None:
            setup_parts.append(
                _minimal_hadoop_conf_script(
                    namenode=_normalize_namenode(self.namenode, default_port=self.hdfs_port),
                    conf_dir=self.minimal_conf_container,
                )
            )
        elif self.hadoop_conf_host:
            env["HADOOP_CONF_DIR"] = self.hadoop_conf_container

        setup = " && ".join(setup_parts)
        prefix = f"{setup} && " if setup else ""

        command = [
            "bash",
            "-lc",
            (
                f"set -euo pipefail; "
                f"{prefix}"
                f"test -d {src!r}; "
                f"hdfs dfs -mkdir -p {hdfs_dst!r}; "
                f"hdfs dfs -put -f {src!r}/. {hdfs_dst!r}/; "
                f"echo copied to {hdfs_dst!r}"
            ),
        ]

        mounts = [context.scratch.to_mount()]
        if self.mount_node_hosts:
            mounts.append(_hosts_mount())
        if self.namenode is None and self.hadoop_conf_host:
            mounts.append(
                {
                    "Type": "bind",
                    "Source": self.hadoop_conf_host,
                    "Target": self.hadoop_conf_container,
                    "ReadOnly": True,
                }
            )

        return CopyJobPlan(image=self.image, command=command, env=env, mounts=mounts)


def copy_strategy_from_config(cfg: dict | str) -> CopyStrategy:
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    strategy_type = cfg.get("type", "hdfs")
    if strategy_type in {"bind", "local", "cp"}:
        return BindCopyStrategy(
            image=cfg.get("image", "alpine:3.20"),
            dest_container_path=cfg.get("dest_container_path", "/dst"),
            scratch_container_path=cfg.get("scratch_container_path"),
        )
    if strategy_type == "hdfs":
        namenode = cfg.get("namenode")
        hadoop_conf_host = cfg.get("hadoop_conf_host")
        if hadoop_conf_host is None and namenode is None:
            hadoop_conf_host = DEFAULT_HADOOP_CONF_HOST

        return HdfsCopyStrategy(
            image=cfg.get("image", "apache/hadoop:3.3.6"),
            namenode=namenode,
            user=cfg.get("user"),
            hadoop_conf_host=hadoop_conf_host,
            hadoop_conf_container=cfg.get(
                "hadoop_conf_container", DEFAULT_HADOOP_CONF_CONTAINER
            ),
            minimal_conf_container=cfg.get(
                "minimal_conf_container", MINIMAL_HADOOP_CONF_CONTAINER
            ),
            mount_node_hosts=cfg.get("mount_node_hosts", True),
            hdfs_port=int(cfg.get("hdfs_port", DEFAULT_HDFS_PORT)),
            extra_env=cfg.get("extra_env", {}),
        )
    raise ValueError(f"unsupported copy strategy: {strategy_type!r}")


def copy_target_from_config(cfg: dict | str) -> CopyTarget:
    if isinstance(cfg, str):
        return CopyTarget(path=cfg)
    return CopyTarget(path=cfg["path"])
