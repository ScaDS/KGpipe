"""
Pipeline and Stage builders for programmatically constructing KG pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..common.models import Data, DataFormat, KgTask, KgPipe, Stage

