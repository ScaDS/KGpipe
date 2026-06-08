#!/usr/bin/env python3
"""
Quick runner for the "Quality Aware Knowledge Graph Pipeline Configurations"
paper mock experiments.

Run from this directory:
  python run_qap_mock.py exp1
  python run_qap_mock.py exp2
  python run_qap_mock.py exp3
  python run_qap_mock.py all
"""

import sys
from pathlib import Path

# Add local experiment src + project src to path
exp_src_path = Path(__file__).parent / "src"
repo_src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(exp_src_path))
sys.path.insert(0, str(repo_src_path))

from qap_mock.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

