from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SCRATCH_HOST = "/local/d1/docker-scratch"
DEFAULT_SCRATCH_CONTAINER = "/local/d1/docker-scratch"


@dataclass(frozen=True)
class ScratchMount:
    """Bind-mount a host scratch directory into the container."""

    host_path: str = DEFAULT_SCRATCH_HOST
    container_path: str = DEFAULT_SCRATCH_CONTAINER

    def to_mount(self) -> dict[str, str]:
        return {
            "Type": "bind",
            "Source": self.host_path,
            "Target": self.container_path,
        }
