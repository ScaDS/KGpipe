from .models import KgTask, DataFormat
from typing import Mapping

registry = []


# TODO: reimport decorators from other packages

# class MetaKg():

#     @staticmethod
#     def activity(func):
#         registry.append(func.__name__)        
#         return func

#     @staticmethod
#     def task():
#         pass

#     @staticmethod
#     def pipeline():
#         pass

#     @staticmethod
#     def tool():
#         pass


# def flextask(inputMapping: Mapping[str, str], outputMapping: Mapping[str, str]):
#     def task_decorator(task_function) -> KgTask:
#         # Convert string format names to DataFormat enum values
#         input_spec = {k: DataFormat(v) for k, v in inputMapping.items()}
#         output_spec = {k: DataFormat(v) for k, v in outputMapping.items()}
#         return KgTask(task_function.__name__, input_spec, output_spec, task_function)
#     return task_decorator

# FOO = "bar"