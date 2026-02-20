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
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster parameters across tools after extraction"
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Skip extraction, only cluster from existing output"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.55,
        help="Cosine distance threshold for clustering (default: 0.55, lower = tighter)"
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
            from kgpipe_llm.common.core import get_client_from_env
            llm_client = get_client_from_env()
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
    
    if not args.cluster_only:
        results = experiment.run(tool_names=args.tool)
        
        # Print extraction summary
        print("\n" + "=" * 60)
        print("Extraction Summary")
        print("=" * 60)
        for name, result in results.items():
            status = "✓" if not result.errors else "⚠"
            print(f"{status} {name}: {len(result.parameters)} parameters from {len(result.sources)} sources")
            if result.errors:
                for err in result.errors[:3]:
                    print(f"    Error: {err}")
    
    # Clustering (after extraction, or standalone with --cluster-only)
    if args.cluster or args.cluster_only:
        print("\n" + "=" * 60)
        print("Clustering Parameters")
        print("=" * 60)
        cluster_result = experiment.cluster_parameters(
            distance_threshold=args.distance_threshold,
        )
        if cluster_result:
            cross_tool = cluster_result.cross_tool_clusters()
            print(f"  Total parameters: {cluster_result.n_parameters}")
            print(f"  Clusters: {cluster_result.n_clusters}")
            print(f"  Cross-tool clusters: {len(cross_tool)}")
            if cross_tool:
                print("\n  Cross-tool clusters:")
                for c in cross_tool[:15]:
                    tools_str = ", ".join(c.tools)
                    print(f"    [{c.cluster_id}] {c.label!r} ({c.size()} params) — tools: {tools_str}")
            print(f"\n  Results saved to: {output_dir / '_clusters.json'}")
            print(f"  Table saved to:   {output_dir / '_parameter_table.csv'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


