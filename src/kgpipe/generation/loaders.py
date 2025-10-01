"""
YAML pipeline loaders and template utilities.
"""

from token import OP
from kgpipe.common import KgPipe, KgTask, Registry, Data

from pydantic import BaseModel, RootModel
from yaml import load, Loader
from pathlib import Path
import json
from typing import Dict, List, Optional

class PipelineDefinition(BaseModel):
    name: str
    description: str
    pipeline: str
    source: str
    target: str
    source_format: str
    target_format: str
    source_schema: str
    target_schema: str

class PipelineCatalog(BaseModel):
    name: str
    description: str
    pipelines: list[PipelineDefinition]


class PipelineConf(BaseModel):
    description: str
    # optional env
    config: Optional[Dict[str, str]] = None
    tasks: List[str]

class PipelineConfs(RootModel[Dict[str, PipelineConf]]):
    pass

"""
{
    "dummy": {
        "description": "Dummy pipeline",
        "tasks": [
            "dummy_task"
        ]
    },
    "dummy_reuse": {
        "description": "Dummy pipeline with reuse of dummy",
        "tasks": [
            "dummy"
        ]
    }
}
"""


def load_pipeline_catalog(path: Path) -> PipelineConfs:
    with open(path, "r") as f:
        return PipelineConfs(**load(f, Loader=Loader))


catalog = PipelineCatalog(
    name="rdf_pipelines",
    description="RDF pipelines",
    pipelines=[]
)

def get_test_data(file_name: str) -> Path:
    return Path(__file__).parent / "test_data" / file_name


def ssp_pipeline(tasks: list[KgTask], target_data: Data, data_dir: str) -> KgPipe:
    pipe = KgPipe(tasks, target_data, data_dir)
    return pipe


def build_from_conf(conf: PipelineConf, target_data: Data, data_dir: str) -> KgPipe:
    tasks = [Registry.get_task(task_name) for task_name in conf.tasks]
    pipe = ssp_pipeline(tasks, target_data, data_dir)
    return pipe

def build_from_yaml(yaml_path: Path):
    pipeline_conf = load_pipeline_catalog(yaml_path)
    pipelines = []
    for pipeline_name, pipeline_conf in pipeline_conf.root.items():
        pass
        # tasks = {task_name: Registry.get("task", task_name) for task_name in pipeline_conf.tasks}
        # pipe = ssp_pipeline(tasks, pipeline_conf.target_data)


# from __future__ import annotations

# import yaml
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Union

# from ..common.models import Data, DataFormat, KgTask, KgPipe, Stage
# from .builders import PipelineBuilder, StageBuilder


# class PipelineLoader:
#     """Loader for pipeline definitions from YAML files."""
    
#     def __init__(self):
#         self.templates = self._load_default_templates()
    
#     def load_pipeline_from_yaml(self, path: Union[str, Path]) -> KgPipe:
#         """Load a pipeline definition from a YAML file."""
#         path = Path(path)
#         if not path.exists():
#             raise FileNotFoundError(f"Pipeline file not found: {path}")
        
#         with open(path, 'r') as f:
#             config = yaml.safe_load(f)
        
#         return self._build_pipeline_from_config(config)
    
#     def load_pipeline_from_string(self, yaml_content: str) -> KgPipe:
#         """Load a pipeline definition from a YAML string."""
#         config = yaml.safe_load(yaml_content)
#         return self._build_pipeline_from_config(config)
    
#     def _build_pipeline_from_config(self, config: Dict[str, Any]) -> KgPipe:
#         """Build a pipeline from a configuration dictionary."""
#         # Validate required fields
#         required_fields = ['name']
#         for field in required_fields:
#             if field not in config:
#                 raise ValueError(f"Missing required field: {field}")
        
#         # Create pipeline builder
#         builder = PipelineBuilder(
#             name=config['name'],
#             description=config.get('description'),
#             parameters=config.get('parameters', {})
#         )
        
#         # Add input data
#         for input_config in config.get('inputs', []):
#             data = self._create_data_from_config(input_config)
#             builder.add_input(data)
        
#         # Add output data
#         for output_config in config.get('outputs', []):
#             data = self._create_data_from_config(output_config)
#             builder.add_output(data)
        
#         # Add stages
#         for stage_config in config.get('stages', []):
#             stage = self._create_stage_from_config(stage_config)
#             builder.add_stage(stage)
        
#         # Validate pipeline
#         errors = builder.validate()
#         if errors:
#             raise ValueError(f"Pipeline validation failed: {'; '.join(errors)}")
        
#         return builder.build()
    
#     def _create_stage_from_config(self, config: Dict[str, Any]) -> Stage:
#         """Create a stage from configuration."""
#         required_fields = ['name']
#         for field in required_fields:
#             if field not in config:
#                 raise ValueError(f"Stage missing required field: {field}")
        
#         stage_builder = StageBuilder(
#             name=config['name'],
#             description=config.get('description'),
#             parameters=config.get('parameters', {}),
#             depends_on=config.get('depends_on', [])
#         )
        
#         # Add inputs
#         for input_config in config.get('inputs', []):
#             if isinstance(input_config, str):
#                 stage_builder.add_input(input_config)
#             else:
#                 data = self._create_data_from_config(input_config)
#                 stage_builder.add_input(data)
        
#         # Add outputs
#         for output_config in config.get('outputs', []):
#             if isinstance(output_config, str):
#                 stage_builder.add_output(output_config)
#             else:
#                 data = self._create_data_from_config(output_config)
#                 stage_builder.add_output(data)
        
#         # Add tasks (this would need to be implemented based on your task system)
#         for task_config in config.get('tasks', []):
#             task = self._create_task_from_config(task_config)
#             stage_builder.add_task(task)
        
#         return stage_builder.build()
    
#     def _create_data_from_config(self, config: Dict[str, Any]) -> Data:
#         """Create a Data object from configuration."""
#         if isinstance(config, str):
#             # Assume it's a path
#             path = Path(config)
#             format_ = self._infer_format_from_path(path)
#             return Data(path, format_)
        
#         path = Path(config['path'])
#         format_str = config.get('format', 'json')
        
#         try:
#             format_ = DataFormat(format_str)
#         except ValueError:
#             format_ = self._infer_format_from_path(path)
        
#         return Data(path, format_)
    
#     def _create_task_from_config(self, config: Dict[str, Any]) -> KgTask:
#         """Create a KgTask from configuration."""
#         # This is a placeholder - you'll need to implement this based on your task system
#         # For now, we'll create a mock task
#         from unittest.mock import Mock
        
#         name = config.get('name', 'mock_task')
#         input_spec = {k: DataFormat(v) for k, v in config.get('input_spec', {}).items()}
#         output_spec = {k: DataFormat(v) for k, v in config.get('output_spec', {}).items()}
        
#         # Create a mock function
#         def mock_function(inputs, outputs):
#             pass
        
#         return KgTask(
#             name=name,
#             input_spec=input_spec,
#             output_spec=output_spec,
#             function=mock_function,
#             description=config.get('description')
#         )
    
#     def _infer_format_from_path(self, path: Path) -> DataFormat:
#         """Infer data format from file path."""
#         suffix = path.suffix.lower()
#         format_map = {
#             '.ttl': DataFormat.TTL,
#             '.nq': DataFormat.NQUADS,
#             '.json': DataFormat.JSON,
#             '.csv': DataFormat.CSV,
#             '.parquet': DataFormat.PARQUET,
#             '.xml': DataFormat.XML,
#             '.rdf': DataFormat.RDF,
#             '.jsonld': DataFormat.JSONLD,
#         }
#         return format_map.get(suffix, DataFormat.JSON)
    
#     def _load_default_templates(self) -> Dict[str, Dict[str, Any]]:
#         """Load default pipeline templates."""
#         return {
#             'simple_etl': {
#                 'name': 'simple_etl',
#                 'description': 'Simple ETL pipeline',
#                 'stages': [
#                     {
#                         'name': 'extract',
#                         'description': 'Extract data from source',
#                         'tasks': []
#                     },
#                     {
#                         'name': 'transform',
#                         'description': 'Transform data',
#                         'tasks': [],
#                         'depends_on': ['extract']
#                     },
#                     {
#                         'name': 'load',
#                         'description': 'Load data to target',
#                         'tasks': [],
#                         'depends_on': ['transform']
#                     }
#                 ]
#             },
#             'entity_linking': {
#                 'name': 'entity_linking',
#                 'description': 'Entity linking pipeline',
#                 'stages': [
#                     {
#                         'name': 'preprocessing',
#                         'description': 'Preprocess text data',
#                         'tasks': []
#                     },
#                     {
#                         'name': 'candidate_generation',
#                         'description': 'Generate entity candidates',
#                         'tasks': [],
#                         'depends_on': ['preprocessing']
#                     },
#                     {
#                         'name': 'ranking',
#                         'description': 'Rank entity candidates',
#                         'tasks': [],
#                         'depends_on': ['candidate_generation']
#                     }
#                 ]
#             }
#         }
    
#     def get_template(self, name: str) -> Dict[str, Any]:
#         """Get a pipeline template by name."""
#         if name not in self.templates:
#             available = list(self.templates.keys())
#             raise ValueError(f"Template '{name}' not found. Available templates: {available}")
        
#         return self.templates[name].copy()
    
#     def list_templates(self) -> List[str]:
#         """List available template names."""
#         return list(self.templates.keys())
    
#     def create_pipeline_from_template(self, template_name: str, **kwargs) -> KgPipe:
#         """Create a pipeline from a template with custom parameters."""
#         template = self.get_template(template_name)
        
#         # Apply custom parameters
#         template.update(kwargs)
        
#         return self._build_pipeline_from_config(template)