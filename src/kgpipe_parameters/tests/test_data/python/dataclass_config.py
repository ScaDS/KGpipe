from dataclasses import dataclass
from typing import Optional

@dataclass
class MatchingConfig:
    """Configuration for matching operations."""
    threshold: float = 0.5
    input_file: str
    output_file: Optional[str] = None
    verbose: bool = False
    max_results: int = 100

