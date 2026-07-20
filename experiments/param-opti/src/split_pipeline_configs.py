#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_fixture(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "samples" not in data:
        raise ValueError(f"Expected dict with 'samples' key in {path}")
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Expected 'samples' to be a list in {path}")
    return data, samples


def task_layout_key(sample: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str(key) for key in (sample.get("task_keys") or []))


def chunk_list(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def split_into_num_parts(
    items: List[Dict[str, Any]], num_parts: int
) -> List[List[Dict[str, Any]]]:
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, got {num_parts}")
    if num_parts > len(items):
        raise ValueError(
            f"num_parts ({num_parts}) cannot exceed number of samples ({len(items)})"
        )

    base_size, remainder = divmod(len(items), num_parts)
    parts: List[List[Dict[str, Any]]] = []
    start = 0
    for index in range(num_parts):
        size = base_size + (1 if index < remainder else 0)
        parts.append(items[start : start + size])
        start += size
    return parts


def split_sequential(
    samples: List[Dict[str, Any]],
    *,
    num_parts: Optional[int],
    max_per_file: Optional[int],
) -> List[List[Dict[str, Any]]]:
    if num_parts is not None and max_per_file is not None:
        raise ValueError("Use only one of --num-parts or --max-per-file for sequential splitting")
    if num_parts is not None:
        return split_into_num_parts(samples, num_parts)
    if max_per_file is not None:
        return chunk_list(samples, max_per_file)
    raise ValueError("Sequential splitting requires --num-parts or --max-per-file")


def split_by_layout(
    samples: List[Dict[str, Any]],
    *,
    max_per_file: Optional[int],
) -> List[List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    layout_order: List[Tuple[str, ...]] = []
    for sample in samples:
        layout = task_layout_key(sample)
        if layout not in grouped:
            grouped[layout] = []
            layout_order.append(layout)
        grouped[layout].append(sample)

    parts: List[List[Dict[str, Any]]] = []
    for layout in layout_order:
        layout_samples = grouped[layout]
        if max_per_file is None:
            parts.append(layout_samples)
        else:
            parts.extend(chunk_list(layout_samples, max_per_file))
    return parts


def output_path(out_dir: Path, stem: str, index: int, total_parts: int) -> Path:
    width = max(2, len(str(total_parts)))
    return out_dir / f"{stem}_{index:0{width}d}.json"


def write_parts(
    *,
    top_level: Dict[str, Any],
    parts: List[List[Dict[str, Any]]],
    out_dir: Path,
    stem: str,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    if not parts:
        raise ValueError("No output parts produced")

    total_parts = len(parts)
    written: List[Dict[str, Any]] = []
    for index, part_samples in enumerate(parts, start=1):
        out_path = output_path(out_dir, stem, index, total_parts)
        out_data = dict(top_level)
        out_data["samples"] = part_samples

        record = {
            "part": index,
            "path": str(out_path),
            "samples": len(part_samples),
        }
        written.append(record)

        if dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split a pipeline configs fixture into numbered sub-files "
            "(e.g. rdf_exhaustive_pipeline_configs_01.json)."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input fixture JSON file")
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for numbered output files",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Output filename stem (default: input filename without extension)",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=None,
        help="Split sequentially into N roughly equal parts",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=None,
        help="Maximum configs per output file",
    )
    parser.add_argument(
        "--by-layout",
        action="store_true",
        help=(
            "Group configs by task layout (task_keys) before splitting. "
            "Without --max-per-file, writes one file per layout."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the split plan without writing files",
    )
    args = parser.parse_args()

    if args.num_parts is None and args.max_per_file is None and not args.by_layout:
        parser.error("Specify --num-parts, --max-per-file, or --by-layout")

    top_level, samples = load_fixture(args.input)
    if not samples:
        raise ValueError(f"No samples found in {args.input}")

    if args.by_layout:
        parts = split_by_layout(samples, max_per_file=args.max_per_file)
    else:
        parts = split_sequential(
            samples,
            num_parts=args.num_parts,
            max_per_file=args.max_per_file,
        )

    stem = args.stem or args.input.stem
    written = write_parts(
        top_level=top_level,
        parts=parts,
        out_dir=args.out_dir,
        stem=stem,
        dry_run=args.dry_run,
    )

    print(
        json.dumps(
            {
                "input_file": str(args.input),
                "out_dir": str(args.out_dir),
                "stem": stem,
                "mode": "layout" if args.by_layout else "sequential",
                "input_samples": len(samples),
                "num_parts": len(parts),
                "dry_run": args.dry_run,
                "parts": written,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
