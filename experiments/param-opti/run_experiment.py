#!/usr/bin/env python3
"""
Quick script to run the parameter extraction experiment.

This script can be run directly from the param-opti directory:
    python run_experiment.py
    python run_experiment.py --tool paris
    python run_experiment.py --no-clone
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Also ensure kgpipe is importable
kgpipe_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(kgpipe_src))

from param_opti.__main__ import main

if __name__ == "__main__":
    sys.exit(main())


