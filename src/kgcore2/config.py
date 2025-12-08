from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Type, TypeVar, Any, Dict, Iterable
from pydantic import BaseModel
from yaml import safe_load
from dotenv import load_dotenv, find_dotenv

# Try to load .env files from multiple locations
# First try the standard .env file
dotenv_path = find_dotenv()
if dotenv_path:
    print(f"[1] Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path, override=True)

# Also try to load from current directory and workspace root
for search_path in [Path.cwd() / ".env"]: #  Path(__file__).parent.parent.parent / ".env"
    if search_path.exists():
        print(f"[2] Loading .env from: {search_path}")
        load_dotenv(search_path, override=True)
        break

# for k, v in os.environ.items():
#     print(k, v)
# Config-specific .env files (e.g., kgpipe.env, moviekg.env) are loaded
# in load_config() from the current working directory


T = TypeVar("T", bound="KGConfig")

class KGConfig(BaseModel):
    """Base config for kg* packages to inherit from."""
    pass


HOME_CONFIG_DIR = Path("~/.kgconf/").expanduser()


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = safe_load(f)  # handles YAML merges/anchors too
    return data or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dicts. Dicts merge recursively; for non-dicts (incl. lists),
    the override wins entirely.
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class ConfigLoader:
    """
    Layered config loader with deep merging.

    Load order (low → high priority):
      1) base:   ~/.kgconf/{name}.yaml (or .yml)
      2) cwd:    ./{name}.yaml (or .yml)
      3) file:   explicit `path` if provided
      4) env:    YAML content from env var {name.upper()}_CONFIG (highest priority)

    Later layers override earlier ones (deep merge).
    """
    def __init__(self, config_name: str):
        self.config_name = config_name or "kgcore"

    def _candidate_paths(self, path: Optional[str | Path]) -> Iterable[Path]:
        # explicit file (if provided)
        explicit = [Path(path)] if path else []

        # support both .yaml and .yml
        base = [
            HOME_CONFIG_DIR / f"{self.config_name}.yaml",
            HOME_CONFIG_DIR / f"{self.config_name}.yml",
        ]
        cwd = [
            Path.cwd() / f"{self.config_name}.yaml",
            Path.cwd() / f"{self.config_name}.yml",
        ]

        # merge order: base → cwd → explicit
        return [p for p in base + cwd + explicit if p is not None]

    def load_config(self, config_class: Type[T], path: Optional[str | Path] = None) -> T:
        merged: Dict[str, Any] = {}
        found_any = False

        # Load from file paths
        for i, p in enumerate(self._candidate_paths(path)):
            d = _read_yaml(p)
            if d:
                found_any = True
                merged = _deep_merge(merged, d)

        # Try to load config-specific .env files (e.g., moviekg.env, kgpipe.env)
        # Search in current directory and kgpipe directory
        workspace_root = Path(__file__).parent.parent.parent
        for env_file_pattern in [
            Path.cwd() / f"{self.config_name}.env",
            workspace_root / f"{self.config_name}.env",
            workspace_root / self.config_name / f"{self.config_name}.env",
            # Also check for moviekg.env in kgpipe directory (special case)
            workspace_root / "kgpipe" / "moviekg.env",
        ]:
            if env_file_pattern.exists():
                print(f"Loading config-specific .env from: {env_file_pattern}")
                load_dotenv(env_file_pattern, override=True)

        # Load from environment variables (highest priority)
        # Get field names from the config class and look for corresponding env vars
        env_prefix = f"{self.config_name.upper()}_"
        env_config = {}
        
        # Check if there's a single {NAME}_CONFIG env var with YAML content
        env_var_name = f"{self.config_name.upper()}_CONFIG"
        env_content = os.environ.get(env_var_name)
        if env_content:
            try:
                d = safe_load(env_content)
                if d:
                    env_config = d
            except Exception as e:
                raise ValueError(f"Failed to parse {env_var_name}: {e}")
        
        # Also check individual env vars for each field in the config class
        for field_name in config_class.model_fields.keys():
            env_var_key = env_prefix + field_name
            env_value = os.environ.get(env_var_key)
            if env_value is not None:
                # Try to parse as YAML for complex types, otherwise use as string
                try:
                    parsed = safe_load(env_value)
                    env_config[field_name] = parsed if isinstance(parsed, dict) else env_value
                except:
                    env_config[field_name] = env_value
        
        if env_config:
            found_any = True
            merged = _deep_merge(merged, env_config)

        # If nothing found, still return defaults from the model
        if not found_any:
            # Check if explicit path was provided but not found
            if path:
                p = Path(path)
                if not p.exists():
                    raise FileNotFoundError(f"Config file not found: {p}")

        return config_class(**merged)

