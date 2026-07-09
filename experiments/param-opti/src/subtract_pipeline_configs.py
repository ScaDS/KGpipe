#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _normalize_bindings(bindings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Ensure stable ordering and stable object shape.
    norm = []  # type: List[Dict[str, Any]]
    for b in bindings:
        norm.append({"parameter": b.get("parameter"), "value": b.get("value")})
    norm.sort(key=lambda x: (str(x.get("parameter")), json.dumps(x.get("value"), sort_keys=True)))
    return norm


def canonical_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    task_keys = sample.get("task_keys") or []
    profiles = sample.get("profiles") or {}

    canon_profiles = {}  # type: Dict[str, Any]
    for profile_key, profile in profiles.items():
        bindings = _normalize_bindings(profile.get("bindings") or [])
        # Prefer the explicit profile_name if present, but don't rely on it exclusively.
        canon_profiles[str(profile_key)] = {
            "profile_name": profile.get("profile_name"),
            "bindings": bindings,
        }

    return {
        "task_keys": sorted(map(str, task_keys)),
        "profiles": {k: canon_profiles[k] for k in sorted(canon_profiles.keys())},
    }


def sample_key(sample: Dict[str, Any]) -> str:
    canon = canonical_sample(sample)
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_fixture(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "samples" not in data:
        raise ValueError(f"Expected dict with 'samples' key in {path}")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Expected 'samples' to be a list in {path}")
    return data, samples


def subtract(
    keep: List[Dict[str, Any]], remove: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int, int]:
    remove_keys = {sample_key(s) for s in remove}
    out = []  # type: List[Dict[str, Any]]
    kept = 0
    dropped = 0
    for s in keep:
        if sample_key(s) in remove_keys:
            dropped += 1
            continue
        out.append(s)
        kept += 1
    return out, kept, dropped


def main() -> int:
    p = argparse.ArgumentParser(
        description="Subtract pipeline config samples between two fixture JSON files."
    )
    p.add_argument("--keep", required=True, type=Path, help="Base fixture (A)")
    p.add_argument("--remove", required=True, type=Path, help="Fixture to subtract (B)")
    p.add_argument("--out", required=True, type=Path, help="Output fixture path (A - B)")
    p.add_argument(
        "--preserve-version",
        action="store_true",
        help="Preserve top-level 'version' from --keep (default: keep entire top-level object and only replace samples).",
    )
    args = p.parse_args()

    keep_data, keep_samples = load_fixture(args.keep)
    _, remove_samples = load_fixture(args.remove)

    out_samples, kept, dropped = subtract(keep_samples, remove_samples)

    # Default behavior: keep the top-level shape of --keep (e.g. version, metadata) and swap samples.
    out_data = {}  # type: Dict[str, Any]
    if args.preserve_version:
        out_data = {"version": keep_data.get("version"), "samples": out_samples}
    else:
        out_data = dict(keep_data)
        out_data["samples"] = out_samples

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "keep_file": str(args.keep),
                "remove_file": str(args.remove),
                "out_file": str(args.out),
                "keep_samples": len(keep_samples),
                "remove_samples": len(remove_samples),
                "out_samples": len(out_samples),
                "dropped_from_keep": dropped,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

