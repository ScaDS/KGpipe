from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# -------------------------
# 1. Config schema
# -------------------------
class KgBackConfig(BaseModel):
    db_url: str = Field(..., description="Database connection URL")
    log_level: str = Field("INFO", description="Logging level")
    cache_dir: Optional[Path] = Field(None, description="Optional cache directory")

    @classmethod
    def from_yaml(cls, path: Path) -> "KgBackConfig":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "KGBACK_") -> dict:
        """
        Collect matching environment variables and return as dict
        """
        mapping = {}
        for field in cls.model_fields:
            env_key = prefix + field.upper()
            if env_key in os.environ:
                mapping[field] = os.environ[env_key]
        return mapping

# -------------------------
# 2. Loader logic
# -------------------------
_DEFAULT_PATH = Path("~/.kgback/config.yaml").expanduser()
_config_singleton: Optional[KgBackConfig] = None


def load_config(path: Optional[str | Path] = None, env_prefix: str = "TOOLNAME_") -> KgBackConfig:
    """
    Layered config loader (singleton pattern):
    base file < given file < env vars
    """
    global _config_singleton
    if _config_singleton:
        return _config_singleton

    # base config
    base_cfg = {}
    if _DEFAULT_PATH.exists():
        base_cfg = yaml.safe_load(_DEFAULT_PATH.read_text()) or {}

    # optional given file
    file_cfg = {}
    if path:
        path = Path(path).expanduser()
        if path.exists():
            file_cfg = yaml.safe_load(path.read_text()) or {}

    # environment overrides
    env_cfg = KgBackConfig.from_env(env_prefix)

    # merge (env > file > base)
    merged = {**base_cfg, **file_cfg, **env_cfg}

    _config_singleton = KgBackConfig(**merged)
    return _config_singleton

def set_config(config: KgBackConfig):
    global _config_singleton
    _config_singleton = config