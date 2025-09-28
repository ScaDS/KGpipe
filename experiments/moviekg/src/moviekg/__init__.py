from kgpipe.execution.config import GLOBAL_STATE

import os
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# TODO rename to context

# class Context(BaseModel):
#     data_dir: str
#     result_dir: str
#     temp_dir: str
#     # seed_kg: KG
#     # target_kg: KG
#     # tasks: list[KgTask]

# def autoload_context() -> Context:
#     data_dir = os.getenv("DATA_DIR")
#     result_dir = os.getenv("RESULT_DIR")
#     temp_dir = os.getenv("TEMP_DIR")
#     if not data_dir:
#         raise ValueError("DATA_DIR environment variable is not set")
#     if not result_dir:
#         raise ValueError("RESULT_DIR environment variable is not set")
#     if not temp_dir:
#         raise ValueError("TEMP_DIR environment variable is not set")
#     return Context(
#         data_dir=data_dir,
#         result_dir=result_dir,
#         temp_dir=temp_dir,
#     )

# CONTEXT = autoload_context()

# def get_data_path(file_name: str, check_exists: bool = True) -> Path:
#     """
#     get data path
#     """
#     path = Path(CONTEXT.data_dir) / file_name
#     if check_exists and not path.exists():
#         raise FileNotFoundError(f"Test data file not found: {path}")
#     return path