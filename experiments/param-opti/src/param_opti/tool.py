"""
Tool definition model for parameter extraction experiments.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


@dataclass
class ToolDefinition:
    """
    Represents a tool to be analyzed for parameter extraction.
    
    A tool is defined by a folder containing:
    - repo.url: Git repository URL
    - cli.txt: (optional) CLI help output
    - readme.md: (optional) README content
    - config.json: (optional) Additional configuration
    """
    name: str
    input_path: Path
    repo_url: Optional[str] = None
    cli_help: Optional[str] = None
    readme_content: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_folder(cls, folder_path: Path) -> "ToolDefinition":
        """
        Load a tool definition from a folder.
        
        Args:
            folder_path: Path to the tool definition folder
            
        Returns:
            ToolDefinition instance
        """
        name = folder_path.name
        
        # Load repo URL
        repo_url = None
        repo_url_file = folder_path / "repo.url"
        if repo_url_file.exists():
            repo_url = repo_url_file.read_text().strip()
        
        # Load CLI help
        cli_help = None
        cli_file = folder_path / "cli.txt"
        if cli_file.exists():
            cli_help = cli_file.read_text()
        
        # Load README
        readme_content = None
        for readme_name in ["readme.md", "README.md", "readme.txt", "README.txt"]:
            readme_file = folder_path / readme_name
            if readme_file.exists():
                readme_content = readme_file.read_text()
                break
        
        # Load config
        config = {}
        config_file = folder_path / "config.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
        
        return cls(
            name=name,
            input_path=folder_path,
            repo_url=repo_url,
            cli_help=cli_help,
            readme_content=readme_content,
            config=config,
        )
    
    def has_repo(self) -> bool:
        """Check if this tool has a repository URL."""
        return self.repo_url is not None and len(self.repo_url) > 0
    
    def has_cli_help(self) -> bool:
        """Check if this tool has CLI help output."""
        return self.cli_help is not None and len(self.cli_help) > 0
    
    def get_language(self) -> Optional[str]:
        """Get the primary language of the tool (from config or auto-detect)."""
        return self.config.get("language")
    
    def __repr__(self) -> str:
        return f"ToolDefinition(name={self.name!r}, repo={self.has_repo()}, cli={self.has_cli_help()})"


