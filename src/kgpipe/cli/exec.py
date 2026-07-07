# Exec pipeline

import click
from pathlib import Path

@click.command()
@click.argument("pipeline", type=str)
@click.option(
    "-c",
    "--config",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=False,
    help="Path to the config file.",
)
@click.option(
    "--config-json",
    type=str,
    required=False,
    help="JSON string of the config.",
)
@click.option(
    "--discover",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Directory to discover additional packages or modules.",
)
@click.option(
    "--mode",
    type=click.Choice(['local', 'docker', 'swarm']),
    default="local",
    help="Execution mode.",
)
def exec_cmd(pipeline: str, config: Path, discover: Path | None):
    """
    Execute a pipeline.
    
    PIPELINE: Name of the pipeline to execute
    CONFIG: Path to the config file
    DISCOVER: Path to the directory to discover additional packages or modules
    """
    
    if mode == "local":
        execute_pipeline_local(pipeline, config, discover)
    elif mode == "docker":
        execute_pipeline_docker(pipeline, config, discover)
    elif mode == "swarm":
        execute_pipeline_swarm(pipeline, config, discover)
    else:
        raise ValueError(f"Invalid mode: {mode}")
# TODO implement