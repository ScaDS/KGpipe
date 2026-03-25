from pydantic import BaseModel, Field
from typing import Optional

class MatchingConfig(BaseModel):
    """Configuration for matching operations."""
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Matching threshold")
    input_file: str = Field(..., description="Input file path (required)")
    output_file: Optional[str] = Field(default=None, description="Output file path")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    max_results: int = Field(default=100, ge=1, description="Maximum number of results")

