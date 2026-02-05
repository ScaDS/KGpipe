"""
Command-line entry point for parameter extraction experiment.

Usage:
    python -m param_opti [--tool TOOL_NAME] [--no-clone] [--use-llm]
"""

import argparse
import sys
from pathlib import Path

from .experiment import ParameterExtractionExperiment


def get_project_root() -> Path:
    """Get the param-opti project root directory."""
    return Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Extract configuration parameters from data integration tools"
    )
    parser.add_argument(
        "--tool", "-t",
        type=str,
        nargs="*",
        help="Specific tool(s) to process (default: all)"
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Skip cloning repositories"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM-based extraction as fallback"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory with tool definitions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=None,
        help="Directory for cloned repositories"
    )
    
    args = parser.parse_args()
    
    # Determine directories
    project_root = get_project_root()
    input_dir = args.input_dir or project_root / "input"
    output_dir = args.output_dir or project_root / "output"
    repos_dir = args.repos_dir or project_root / "repos"
    
    # Initialize LLM client if requested
    llm_client = None
    if args.use_llm:
        try:
            from kgpipe_llm.common.core import LLMClient
            llm_client = LLMClient()
            print("LLM client initialized")
        except ImportError:
            print("Warning: kgpipe_llm not available, proceeding without LLM")
    
    # Create and run experiment
    experiment = ParameterExtractionExperiment(
        input_dir=input_dir,
        output_dir=output_dir,
        repos_dir=repos_dir,
        clone_repos=not args.no_clone,
        use_llm=args.use_llm,
        llm_client=llm_client,
    )
    
    results = experiment.run(tool_names=args.tool)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if not result.errors else "⚠"
        print(f"{status} {name}: {len(result.parameters)} parameters from {len(result.sources)} sources")
        if result.errors:
            for err in result.errors[:3]:  # Show first 3 errors
                print(f"    Error: {err}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


