"""
Main experiment runner for parameter extraction.
"""

import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from .tool import ToolDefinition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionSource:
    """Represents a source that was used for extraction."""
    source_type: str  # cli, python, docker, readme, etc.
    file_path: Optional[str] = None
    content_preview: Optional[str] = None
    parameters_count: int = 0


@dataclass 
class ToolExtractionResult:
    """Result of parameter extraction for a single tool."""
    tool_name: str
    timestamp: str
    sources: List[ExtractionSource] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "sources": [asdict(s) for s in self.sources],
            "parameters": self.parameters,
            "errors": self.errors,
            "metadata": self.metadata,
            "summary": {
                "total_parameters": len(self.parameters),
                "total_sources": len(self.sources),
                "total_errors": len(self.errors),
            }
        }


class ParameterExtractionExperiment:
    """
    Main experiment class for extracting parameters from tools.
    
    This class orchestrates the parameter extraction process:
    1. Discovers tools from input folder
    2. Clones repositories if needed
    3. Applies extractors to various sources (CLI, Python, Docker, etc.)
    4. Aggregates and saves results
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        repos_dir: Path,
        clone_repos: bool = True,
        use_llm: bool = False,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the experiment.
        
        Args:
            input_dir: Directory containing tool definitions
            output_dir: Directory for output results
            repos_dir: Directory for cloned repositories
            clone_repos: Whether to clone repositories
            use_llm: Whether to use LLM-based extraction as fallback
            llm_client: Optional LLM client instance
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.repos_dir = Path(repos_dir)
        self.clone_repos = clone_repos
        self.use_llm = use_llm
        self.llm_client = llm_client
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize miner lazily
        self._miner = None
    
    @property
    def miner(self):
        """Lazy initialization of ParameterMiner."""
        if self._miner is None:
            from kgpipe_parameters.extraction import ParameterMiner
            self._miner = ParameterMiner(llm_client=self.llm_client)
        return self._miner
    
    def discover_tools(self) -> List[ToolDefinition]:
        """
        Discover all tool definitions in the input directory.
        
        Returns:
            List of ToolDefinition instances
        """
        tools = []
        for folder in sorted(self.input_dir.iterdir()):
            if folder.is_dir() and not folder.name.startswith("."):
                try:
                    tool = ToolDefinition.from_folder(folder)
                    tools.append(tool)
                    logger.info(f"Discovered tool: {tool}")
                except Exception as e:
                    logger.warning(f"Failed to load tool from {folder}: {e}")
        
        logger.info(f"Discovered {len(tools)} tools")
        return tools
    
    def clone_repository(self, tool: ToolDefinition) -> Optional[Path]:
        """
        Clone a tool's repository if not already present.
        
        Args:
            tool: Tool definition with repo URL
            
        Returns:
            Path to the cloned repository, or None if failed
        """
        if not tool.has_repo():
            logger.warning(f"No repository URL for {tool.name}")
            return None
        
        repo_path = self.repos_dir / tool.name
        
        if repo_path.exists():
            logger.info(f"Repository already exists: {repo_path}")
            return repo_path
        
        logger.info(f"Cloning {tool.repo_url} to {repo_path}")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", tool.repo_url, str(repo_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode == 0:
                logger.info(f"Successfully cloned {tool.name}")
                return repo_path
            else:
                logger.error(f"Git clone failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error(f"Git clone timed out for {tool.name}")
            return None
        except Exception as e:
            logger.error(f"Failed to clone {tool.name}: {e}")
            return None
    
    def extract_from_cli(self, tool: ToolDefinition) -> Optional[Dict[str, Any]]:
        """
        Extract parameters from CLI help output.
        
        Args:
            tool: Tool definition with CLI help
            
        Returns:
            Extraction result dictionary, or None if no CLI help
        """
        if not tool.has_cli_help():
            return None
        
        logger.info(f"Extracting from CLI help for {tool.name}")
        from kgpipe_parameters.extraction import SourceType
        
        result = self.miner.extract_parameters(
            source=tool.cli_help,
            source_type=SourceType.CLI,
            tool_name=tool.name,
        )
        
        return {
            "source_type": "cli",
            "source_file": str(tool.input_path / "cli.txt"),
            "result": json.loads(result.model_dump_json()),
        }
    
    def extract_from_readme(self, tool: ToolDefinition) -> Optional[Dict[str, Any]]:
        """
        Extract parameters from a README bundled with the tool input definition.
        
        Args:
            tool: Tool definition with readme_content
            
        Returns:
            Extraction result dictionary, or None if no README content
        """
        if not tool.readme_content:
            return None
        
        logger.info(f"Extracting from bundled README for {tool.name}")
        from kgpipe_parameters.extraction import SourceType
        
        result = self.miner.extract_parameters(
            source=tool.readme_content,
            source_type=SourceType.README,
            tool_name=tool.name,
        )
        
        return {
            "source_type": "readme",
            "source_file": str(tool.input_path / "readme.md"),
            "result": json.loads(result.model_dump_json()),
        }
    
    def extract_from_repo(self, tool: ToolDefinition, repo_path: Path) -> List[Dict[str, Any]]:
        """
        Extract parameters from repository files.
        
        Scans Python, Java, .properties, .xml, Docker, and README/doc files.
        A keyword-based chunk filter is applied first so that only files
        containing parameter-signal keywords are sent to the extractors,
        preventing noise from irrelevant source files.
        
        Args:
            tool: Tool definition
            repo_path: Path to cloned repository
            
        Returns:
            List of extraction result dictionaries
        """
        from kgpipe_parameters.extraction import SourceType
        from kgpipe_parameters.extraction.chunk_filter import has_parameter_signals, score_chunk
        
        results = []
        
        # ------------------------------------------------------------------
        # Helper: prioritize files whose names suggest config / CLI / params
        # ------------------------------------------------------------------
        priority_patterns = [
            "main", "cli", "config", "settings", "args", "params", "options",
            "__main__", "run", "train", "evaluate", "application", "setup",
        ]
        
        def priority_score(path: Path) -> int:
            name = path.stem.lower()
            for i, pattern in enumerate(priority_patterns):
                if pattern in name:
                    return i
            return len(priority_patterns)
        
        def _is_test_file(path: Path) -> bool:
            """Return True for test / example files we want to skip."""
            low = str(path).lower()
            return any(s in low for s in ["test", "/example", "/demo", "/sample"])
        
        # ==================================================================
        # 1. Python files
        # ==================================================================
        python_files = sorted(repo_path.rglob("*.py"), key=priority_score)
        logger.info(f"Found {len(python_files)} Python files in {tool.name}")
        
        accepted_py = 0
        for py_file in python_files:
            if accepted_py >= 20:
                break
            try:
                content = py_file.read_text(errors="ignore")
                if len(content) < 100 or _is_test_file(py_file):
                    continue
                
                # ── Keyword chunk filter ──
                if not has_parameter_signals(content, file_path=str(py_file)):
                    logger.debug(f"  Skipped (no param signals): {py_file.name}")
                    continue
                
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.PYTHON_LIB,
                    tool_name=f"{tool.name}/{py_file.name}",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "python",
                        "source_file": str(py_file.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {py_file.name}")
                accepted_py += 1
            except Exception as e:
                logger.warning(f"  Failed to process {py_file}: {e}")
        
        # ==================================================================
        # 2. Java files
        # ==================================================================
        java_files = sorted(repo_path.rglob("*.java"), key=priority_score)
        logger.info(f"Found {len(java_files)} Java files in {tool.name}")
        
        accepted_java = 0
        for java_file in java_files:
            if accepted_java >= 20:
                break
            try:
                content = java_file.read_text(errors="ignore")
                if len(content) < 100 or _is_test_file(java_file):
                    continue
                
                # ── Keyword chunk filter ──
                if not has_parameter_signals(content, file_path=str(java_file)):
                    logger.debug(f"  Skipped (no param signals): {java_file.name}")
                    continue
                
                # Java config files are best handled by the README extractor
                # (it picks up flag patterns, key-value pairs, etc.)
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.README,
                    tool_name=f"{tool.name}/{java_file.name}",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "java",
                        "source_file": str(java_file.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {java_file.name}")
                accepted_java += 1
            except Exception as e:
                logger.warning(f"  Failed to process {java_file}: {e}")
        
        # ==================================================================
        # 3. .properties files (Java native config format)
        # ==================================================================
        properties_files = list(repo_path.rglob("*.properties"))
        logger.info(f"Found {len(properties_files)} .properties files in {tool.name}")
        
        for prop_file in properties_files[:15]:
            try:
                content = prop_file.read_text(errors="ignore")
                if len(content) < 10 or _is_test_file(prop_file):
                    continue
                
                # .properties files are inherently config — always relevant
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.README,  # kv-pair patterns work well
                    tool_name=f"{tool.name}/{prop_file.name}",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "properties",
                        "source_file": str(prop_file.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {prop_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to process {prop_file}: {e}")
        
        # ==================================================================
        # 4. XML config files
        # ==================================================================
        xml_files = list(repo_path.rglob("*.xml"))
        # Only keep files whose names suggest config, not build scripts
        _xml_config_hints = [
            "config", "setting", "param", "property", "application",
            "persistence", "context", "bean",
        ]
        xml_files = [
            f for f in xml_files
            if any(h in f.stem.lower() for h in _xml_config_hints)
            or has_parameter_signals(
                f.read_text(errors="ignore")[:2000],
                file_path=str(f),
            )
        ]
        logger.info(f"Found {len(xml_files)} XML config files in {tool.name}")
        
        for xml_file in xml_files[:10]:
            try:
                content = xml_file.read_text(errors="ignore")
                if len(content) < 30 or _is_test_file(xml_file):
                    continue
                
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.README,
                    tool_name=f"{tool.name}/{xml_file.name}",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "xml",
                        "source_file": str(xml_file.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {xml_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to process {xml_file}: {e}")
        
        # ==================================================================
        # 5. Dockerfiles
        # ==================================================================
        for dockerfile in repo_path.rglob("Dockerfile*"):
            try:
                content = dockerfile.read_text(errors="ignore")
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.DOCKER,
                    tool_name=f"{tool.name}/Dockerfile",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "docker",
                        "source_file": str(dockerfile.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {dockerfile.name}")
            except Exception as e:
                logger.warning(f"  Failed to process {dockerfile}: {e}")
        
        # ==================================================================
        # 6. docker-compose files
        # ==================================================================
        for compose_file in repo_path.rglob("docker-compose*.y*ml"):
            try:
                content = compose_file.read_text(errors="ignore")
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.DOCKER,
                    tool_name=f"{tool.name}/docker-compose",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "docker",
                        "source_file": str(compose_file.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {compose_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to process {compose_file}: {e}")
        
        # ==================================================================
        # 7. README and documentation files
        # ==================================================================
        readme_patterns = ["README*", "readme*", "INSTALL*", "USAGE*", "CONFIGURATION*"]
        doc_dirs = ["doc", "docs", "documentation"]
        
        readme_files: List[Path] = []
        for pattern in readme_patterns:
            readme_files.extend(repo_path.glob(pattern))
        # Also pick up .md files scattered in the repo root (e.g. RunPARIS.md)
        readme_files.extend(repo_path.glob("*.md"))
        for doc_dir_name in doc_dirs:
            doc_dir = repo_path / doc_dir_name
            if doc_dir.is_dir():
                readme_files.extend(doc_dir.rglob("*.md"))
                readme_files.extend(doc_dir.rglob("*.txt"))
                readme_files.extend(doc_dir.rglob("*.rst"))
        
        # Deduplicate while preserving order
        seen_readme: set = set()
        unique_readmes: List[Path] = []
        for f in readme_files:
            if f.resolve() not in seen_readme and f.is_file():
                seen_readme.add(f.resolve())
                unique_readmes.append(f)
        
        logger.info(f"Found {len(unique_readmes)} README/doc files in {tool.name}")
        
        for readme_file in unique_readmes[:15]:
            try:
                content = readme_file.read_text(errors="ignore")
                if len(content) < 50:
                    continue
                
                # ── Keyword chunk filter for docs ──
                if not has_parameter_signals(content, file_path=str(readme_file), threshold=1):
                    logger.debug(f"  Skipped (no param signals): {readme_file.name}")
                    continue
                
                result = self.miner.extract_parameters(
                    source=content,
                    source_type=SourceType.README,
                    tool_name=f"{tool.name}/{readme_file.name}",
                )
                
                if result.parameters:
                    results.append({
                        "source_type": "readme",
                        "source_file": str(readme_file.relative_to(repo_path)),
                        "result": json.loads(result.model_dump_json()),
                    })
                    logger.info(f"  Extracted {len(result.parameters)} params from {readme_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to process {readme_file}: {e}")
        
        return results
    
    def process_tool(self, tool: ToolDefinition) -> ToolExtractionResult:
        """
        Process a single tool and extract all parameters.
        
        Args:
            tool: Tool definition to process
            
        Returns:
            ToolExtractionResult with all extracted parameters
        """
        logger.info(f"Processing tool: {tool.name}")
        
        result = ToolExtractionResult(
            tool_name=tool.name,
            timestamp=datetime.now().isoformat(),
            metadata={
                "repo_url": tool.repo_url,
                "has_cli_help": tool.has_cli_help(),
                "config": tool.config,
            }
        )
        
        # Extract from CLI help
        if tool.has_cli_help():
            try:
                cli_result = self.extract_from_cli(tool)
                if cli_result:
                    params = cli_result["result"].get("parameters", [])
                    result.sources.append(ExtractionSource(
                        source_type="cli",
                        file_path=cli_result["source_file"],
                        content_preview=tool.cli_help[:200] if tool.cli_help else None,
                        parameters_count=len(params),
                    ))
                    for p in params:
                        p["_source"] = "cli"
                        result.parameters.append(p)
            except Exception as e:
                result.errors.append(f"CLI extraction failed: {str(e)}")
                logger.error(f"CLI extraction failed for {tool.name}: {e}")
        
        # Extract from bundled README
        if tool.readme_content:
            try:
                readme_result = self.extract_from_readme(tool)
                if readme_result:
                    params = readme_result["result"].get("parameters", [])
                    result.sources.append(ExtractionSource(
                        source_type="readme",
                        file_path=readme_result["source_file"],
                        content_preview=tool.readme_content[:200] if tool.readme_content else None,
                        parameters_count=len(params),
                    ))
                    for p in params:
                        p["_source"] = "readme"
                        result.parameters.append(p)
            except Exception as e:
                result.errors.append(f"README extraction failed: {str(e)}")
                logger.error(f"README extraction failed for {tool.name}: {e}")
        
        # Clone (if requested) and extract from repository
        if tool.has_repo():
            repo_path = self.repos_dir / tool.name
            if self.clone_repos:
                repo_path = self.clone_repository(tool)
            if repo_path and repo_path.exists():
                try:
                    repo_results = self.extract_from_repo(tool, repo_path)
                    for r in repo_results:
                        params = r["result"].get("parameters", [])
                        result.sources.append(ExtractionSource(
                            source_type=r["source_type"],
                            file_path=r["source_file"],
                            parameters_count=len(params),
                        ))
                        for p in params:
                            p["_source"] = f"{r['source_type']}:{r['source_file']}"
                            result.parameters.append(p)
                except Exception as e:
                    result.errors.append(f"Repository extraction failed: {str(e)}")
                    logger.error(f"Repository extraction failed for {tool.name}: {e}")
        
        logger.info(f"Completed {tool.name}: {len(result.parameters)} parameters from {len(result.sources)} sources")
        return result
    
    def save_result(self, result: ToolExtractionResult) -> Path:
        """
        Save extraction result to output directory.
        
        Args:
            result: Extraction result to save
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"{result.tool_name}.json"
        
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved result to {output_file}")
        return output_file
    
    def run(self, tool_names: Optional[List[str]] = None) -> Dict[str, ToolExtractionResult]:
        """
        Run the experiment for all or selected tools.
        
        Args:
            tool_names: Optional list of tool names to process (all if None)
            
        Returns:
            Dictionary mapping tool names to their extraction results
        """
        logger.info("=" * 60)
        logger.info("Starting Parameter Extraction Experiment")
        logger.info("=" * 60)
        
        # Discover tools
        tools = self.discover_tools()
        
        # Filter if specific tools requested
        if tool_names:
            tools = [t for t in tools if t.name in tool_names]
            logger.info(f"Filtered to {len(tools)} tools: {[t.name for t in tools]}")
        
        # Process each tool
        results = {}
        for tool in tools:
            try:
                result = self.process_tool(tool)
                self.save_result(result)
                results[tool.name] = result
            except Exception as e:
                logger.error(f"Failed to process {tool.name}: {e}")
                results[tool.name] = ToolExtractionResult(
                    tool_name=tool.name,
                    timestamp=datetime.now().isoformat(),
                    errors=[str(e)],
                )
        
        # Generate summary
        self._generate_summary(results)
        
        logger.info("=" * 60)
        logger.info("Experiment Complete")
        logger.info("=" * 60)
        
        return results
    
    def cluster_parameters(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        distance_threshold: float = 0.55,
    ) -> Optional[Any]:
        """
        Cluster extracted parameters across all tools using sentence-transformer
        embeddings and agglomerative clustering.

        This reads the per-tool JSON files already written to ``output_dir``,
        embeds every parameter, and groups similar ones together.

        Args:
            model_name: Sentence-transformer model identifier.
            distance_threshold: Max cosine distance for merging (lower = tighter).

        Returns:
            A ClusteringResult, or None if no parameters were found.
        """
        from kgpipe_parameters.clustering import ParameterClusterer

        clusterer = ParameterClusterer(
            model_name=model_name,
            distance_threshold=distance_threshold,
        )

        result = clusterer.cluster_from_output_dir(self.output_dir)

        if result.n_clusters == 0:
            logger.warning("Clustering produced 0 clusters")
            return result

        # Persist results
        clusterer.save_result(result, self.output_dir / "_clusters.json")
        clusterer.save_table(result, self.output_dir / "_parameter_table.csv")

        # Log summary
        cross_tool = result.cross_tool_clusters()
        logger.info(
            "Clustering: %d parameters → %d clusters (%d cross-tool)",
            result.n_parameters,
            result.n_clusters,
            len(cross_tool),
        )
        return result

    def _generate_summary(self, results: Dict[str, ToolExtractionResult]) -> None:
        """Generate and save experiment summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tools": len(results),
            "tools": {}
        }
        
        total_params = 0
        total_sources = 0
        total_errors = 0
        
        for name, result in results.items():
            summary["tools"][name] = {
                "parameters": len(result.parameters),
                "sources": len(result.sources),
                "errors": len(result.errors),
            }
            total_params += len(result.parameters)
            total_sources += len(result.sources)
            total_errors += len(result.errors)
        
        summary["totals"] = {
            "parameters": total_params,
            "sources": total_sources,
            "errors": total_errors,
        }
        
        summary_file = self.output_dir / "_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary: {total_params} parameters from {total_sources} sources ({total_errors} errors)")


